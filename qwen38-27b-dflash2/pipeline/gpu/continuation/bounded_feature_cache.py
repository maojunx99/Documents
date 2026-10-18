"""Quota and train-progress bounds for the paired feature cache.

This module is integration code, not an independent teacher/training launcher.
Reservations persist on failure for diagnosis; no unknown cache is evicted.
"""
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import torch
from shared_feature_cache import SharedFeatureCache


class CacheQuota:
    def __init__(self,root,*,limit_bytes=96*1024**3,timeout=600):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'quota.sqlite';self.limit=int(limit_bytes);self.timeout=timeout
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config(limit_bytes INTEGER NOT NULL)')
            rows=db.execute('SELECT limit_bytes FROM config').fetchall()
            if rows:assert rows==[(self.limit,)], 'Cache quota changed in an active namespace'
            else:db.execute('INSERT INTO config VALUES(?)',(self.limit,))
            db.execute('CREATE TABLE IF NOT EXISTS reservations(key TEXT PRIMARY KEY,bytes INTEGER NOT NULL,created REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY,value INTEGER NOT NULL)')

    @contextlib.contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        try:
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()

    def reserve(self,key,nbytes):
        nbytes=int(nbytes)
        if not 0<nbytes<=self.limit:raise ValueError('Single feature reservation exceeds cache cap')
        deadline=time.monotonic()+self.timeout
        while True:
            with self.connect() as db:
                prior=db.execute('SELECT bytes FROM reservations WHERE key=?',(key,)).fetchone()
                if prior:raise RuntimeError('Reservation already exists; diagnose previous producer '+key)
                used=db.execute('SELECT COALESCE(SUM(bytes),0) FROM reservations').fetchone()[0]
                if used+nbytes<=self.limit:
                    db.execute('INSERT INTO reservations VALUES(?,?,?)',(key,nbytes,time.time()))
                    db.execute("INSERT INTO counters VALUES('high_water_reserved_bytes',?) ON CONFLICT(name) DO UPDATE SET value=MAX(value,excluded.value)",(used+nbytes,))
                    return
            if time.monotonic()>deadline:raise TimeoutError('Shared feature quota backpressure timeout; no cache evicted')
            time.sleep(.1)

    def release_after_dual_ack(self,key):
        with self.connect() as db:
            db.execute('DELETE FROM reservations WHERE key=?',(key,))

    def snapshot(self):
        with self.connect() as db:
            return dict(limit_bytes=self.limit,reserved_bytes=db.execute('SELECT COALESCE(SUM(bytes),0) FROM reservations').fetchone()[0],pending_keys=db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0],counters=dict(db.execute('SELECT name,value FROM counters')))


class BoundedSharedFeatureCache(SharedFeatureCache):
    def __init__(self,root,arm,*,arms=('overlap','sampled_ce'),timeout=600,limit_bytes=96*1024**3):
        super().__init__(root,arm,arms=arms,timeout=timeout)
        self.quota=CacheQuota(self.root,limit_bytes=limit_bytes,timeout=timeout)

    def fetch(self,key,input_ids,producer,*,hidden_shape):
        hidden_shape=tuple(map(int,hidden_shape))
        if not hidden_shape or hidden_shape[0]!=input_ids.numel():raise ValueError('Expected per-sequence hidden shape')
        elements=1
        for n in hidden_shape:
            if n<=0:raise ValueError('Invalid feature dimension')
            elements*=n
        # BF16 features plus int64 ids, conservatively allow a second serialized
        # copy and 1MiB container overhead. Producer's native transport is separately bounded.
        reservation=2*(elements*2+input_ids.numel()*8)+1024**2
        def bounded_producer():
            self.quota.reserve(key,reservation)
            tensors=producer();h=tensors['hidden_states']
            if tuple(h.shape)!=hidden_shape or h.dtype!=torch.bfloat16:
                raise ValueError('Teacher hidden shape/dtype differs from reserved BF16 layout')
            return tensors
        result=super().fetch(key,input_ids,bounded_producer)
        digest=hashlib.sha256(key.encode()).hexdigest();directory=self.root/digest[:2]/digest
        if (directory/'deleted.json').exists():
            if not all((directory/('ack_'+arm+'.json')).exists() for arm in self.arms):raise RuntimeError('Missing dual ack')
            if (directory/'tensors.pt').exists():raise RuntimeError('Payload remains after deletion marker')
            self.quota.release_after_dual_ack(key)
        return result


class PairedProgressGate:
    """Trainer-side bound; DataLoader prefetch allowance is separate from this.

    Each rank publishes completed optimizer boundaries. No automatic recovery,
    rank skipping, or changing the bound is allowed.
    """
    def __init__(self,root,arm,rank,*,arms=('overlap','sampled_ce'),world_size=2,max_ahead_updates=8,timeout=600):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        if arm not in arms or not 0<=rank<world_size:raise ValueError('Unknown consumer rank')
        self.arm=arm;self.rank=rank;self.arms=tuple(arms);self.world_size=world_size
        self.max_ahead=int(max_ahead_updates);self.timeout=timeout

    def publish(self,completed):
        path=self.root/f'{self.arm}_rank{self.rank}.json'
        if path.exists():
            prior=json.loads(path.read_text())['completed_optimizer_steps']
            if completed<prior:raise RuntimeError('Progress went backwards; explicit restore protocol required')
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(dict(completed_optimizer_steps=int(completed),time=time.time())));temp.replace(path)

    def wait_before_update(self,next_step):
        deadline=time.monotonic()+self.timeout
        while True:
            values=[]
            for arm in self.arms:
                for rank in range(self.world_size):
                    path=self.root/f'{arm}_rank{rank}.json'
                    values.append(json.loads(path.read_text())['completed_optimizer_steps'] if path.exists() else 0)
            if next_step<=min(values)+self.max_ahead:return
            if time.monotonic()>deadline:raise TimeoutError('Paired trainer progress skew timeout; peer may have failed')
            time.sleep(.25)
