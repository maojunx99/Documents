"""Paired feature cache: one producer, verified copies, two acknowledgements.

Keys identify occurrences (phase/epoch/sample), not just dataset row numbers.
No mmap views escape the lock. An artifact is unlinked only after both arms
have loaded and verified it. Metadata/ack/failure evidence remains retained.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import time
import torch


def tensor_sha(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


class SharedFeatureCache:
    def __init__(self,root,arm,arms=('overlap','sampled_ce'),timeout=600):
        self.root=Path(root).resolve();self.arm=arm;self.arms=tuple(arms);self.timeout=timeout
        if arm not in self.arms or len(set(self.arms))!=2:raise ValueError('Expected two named arms')
        self.root.mkdir(parents=True,exist_ok=True)

    def fetch(self,key,input_ids,producer):
        digest=hashlib.sha256(key.encode()).hexdigest();d=self.root/digest[:2]/digest;d.mkdir(parents=True,exist_ok=True)
        lock=(d/'lock').open('a+b');deadline=time.monotonic()+self.timeout
        try:
            while True:
                try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise TimeoutError('Paired feature lock timeout '+digest)
                    time.sleep(.1)
            if (d/'failure.json').exists():raise RuntimeError('Previous feature production failed: '+digest)
            metadata=d/'metadata.json';payload=d/'tensors.pt'
            if not metadata.exists():
                try:
                    tensors=producer()
                    if set(tensors)!={'hidden_states','token_ids'}:raise ValueError('Unexpected teacher fields')
                    if not torch.equal(tensors['token_ids'],input_ids):raise ValueError('Teacher token mismatch')
                    if not torch.isfinite(tensors['hidden_states']).all():raise ValueError('Nonfinite teacher features')
                    tensors={k:v.detach().cpu().clone() for k,v in tensors.items()}
                    torch.save(tensors,d/'tensors.pt.tmp');(d/'tensors.pt.tmp').replace(payload)
                    m=dict(key=key,arms=self.arms,created=time.time(),sha256={k:tensor_sha(v) for k,v in tensors.items()},shape={k:list(v.shape) for k,v in tensors.items()},dtype={k:str(v.dtype) for k,v in tensors.items()})
                    (d/'metadata.json.tmp').write_text(json.dumps(m));(d/'metadata.json.tmp').replace(metadata)
                except BaseException as e:
                    (d/'failure.json').write_text(json.dumps(dict(key=key,error=repr(e),time=time.time())));raise
            m=json.loads(metadata.read_text())
            if m['key']!=key or tuple(m['arms'])!=self.arms:raise RuntimeError('Feature cache namespace mismatch')
            if (d/('ack_'+self.arm+'.json')).exists():raise RuntimeError('Duplicate occurrence consumer '+self.arm+' '+key)
            tensors=torch.load(payload,map_location='cpu',weights_only=True,mmap=False)
            if {k:tensor_sha(v) for k,v in tensors.items()}!=m['sha256']:raise RuntimeError('Feature checksum changed')
            if not torch.equal(tensors['token_ids'],input_ids):raise RuntimeError('Consumer token mismatch')
            (d/('ack_'+self.arm+'.json')).write_text(json.dumps(dict(time=time.time(),sha256=m['sha256'])))
            if all((d/('ack_'+a+'.json')).exists() for a in self.arms):
                assert payload.resolve().parent==d.resolve() and d.is_relative_to(self.root)
                payload.unlink();(d/'deleted.json').write_text(json.dumps(dict(time=time.time(),reason='both verified consumers loaded private tensors')))
            return tensors,m
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
