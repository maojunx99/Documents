"""Four-pack update and committed FP32 trainer-state checkpoints.

This is an integration component, not a launcher. The native forward, locked
sampler, paired feature producer and inference checkpoint export remain external.
Do not treat a CPU component preflight as the required 20-step GPU smoke.
"""
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from training_primitives import FP32MasterAdamW


@dataclass
class Cursor:
    completed_optimizer_steps: int = 0
    next_microbatch: int = 0
    global_input_tokens: int = 0
    global_valid_positions: int = 0


@dataclass
class ForwardNumerator:
    # All native weighted numerators, including alpha, share UNWEIGHTED count.
    numerator: torch.Tensor
    valid_positions: torch.Tensor
    nonpadding_input_tokens: int


def cosine_schedule(optimizer, *, total_updates, warmup_updates):
    if not 0 < warmup_updates < total_updates:
        raise ValueError('Freeze positive warmup and total update counts first')
    def factor(completed):
        if completed < warmup_updates:
            return completed / warmup_updates
        ratio=(completed-warmup_updates)/(total_updates-warmup_updates)
        return .5*(1+math.cos(math.pi*min(1.,max(0.,ratio))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save_file(path, value, *, tensor=False):
    path=Path(path)
    with path.open('xb') as f:
        if tensor:torch.save(value,f)
        else:f.write((json.dumps(value,sort_keys=True,indent=2)+'\n').encode())
        f.flush();os.fsync(f.fileno())
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':digest_file(path)}


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def validate_rng(state):
    if len(state['torch_cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA RNG topology changed')
    # Validate without advancing or replacing this process's generators.
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    for saved in state['torch_cuda']:
        if saved.device.type!='cpu' or saved.dtype!=torch.uint8 or saved.ndim!=1:
            raise ValueError('Invalid CUDA RNG payload')


def restore_rng(state):
    validate_rng(state)
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


class PairedTrainingEngine:
    def __init__(self, model, master, scheduler, *, total_updates, input_token_cap,
                 fingerprints, distributed=False):
        self.model=model;self.master=master;self.scheduler=scheduler
        self.total_updates=int(total_updates);self.input_token_cap=int(input_token_cap)
        self.fingerprints=dict(fingerprints);self.distributed=bool(distributed)
        self.rank=dist.get_rank() if distributed else 0
        self.world=dist.get_world_size() if distributed else 1
        if distributed and not hasattr(model,'no_sync'):raise TypeError('DDP model required')
        if master.expected_microbatches!=4:raise ValueError('Locked four-microbatch policy')
        if scheduler.optimizer is not master.optimizer:raise ValueError('Wrong scheduler optimizer')
        if not self.fingerprints.get('schedule_sha256'):raise ValueError('Missing locked schedule')
        self.cursor=Cursor();self.failed=False

    def gather(self,value):
        if not self.distributed:return [value]
        values=[None]*self.world;dist.all_gather_object(values,value);return values

    def check_errors(self,error):
        errors=self.gather(error)
        if any(e is not None for e in errors):
            self.failed=True
            raise RuntimeError('Rank validation failed: '+repr(errors))

    def update(self,batches,forward,*,expected_global_input_tokens):
        """Exactly four no_sync forward/backwards, one reduction/clip/update.

        `forward(model,batch)` returns unnormalized loss and native valid count.
        Both arms must independently validate their same sealed sampler entries.
        Counts are checked BEFORE optimizer mutation. A failure poisons this
        instance: no silently retrying a partial gradient window.
        """
        error=None
        if self.failed:error='Previous window failed; diagnose, do not reuse'
        elif len(batches)!=4:error='Incomplete optimizer window'
        elif self.master.microbatches:error='Uncleared partial gradients'
        elif self.cursor.completed_optimizer_steps>=self.total_updates:error='Update budget exceeded'
        elif self.cursor.global_input_tokens+expected_global_input_tokens>self.input_token_cap:error='Token budget exceeded'
        self.check_errors(error)
        local_input=0;local_valid=0
        for batch in batches:
            ctx=self.model.no_sync() if self.distributed else contextlib.nullcontext()
            with ctx:
                result=None;error=None
                try:
                    result=forward(self.model,batch)
                    n=result.numerator;d=result.valid_positions
                    if n.ndim or d.ndim or not n.requires_grad:raise ValueError('Scalar differentiable numerator and scalar count required')
                    if not torch.isfinite(n).item() or not torch.isfinite(d).item():raise FloatingPointError('Nonfinite loss/count')
                    if d.item()<=0 or d.item()!=int(d.item()):raise ValueError('Invalid native valid-position count')
                    if result.nonpadding_input_tokens<=0:raise ValueError('Empty input')
                except Exception as e:error=f'{type(e).__name__}: {e}'
                self.check_errors(error)
                error=None
                try:
                    result.numerator.backward()
                    self.master.capture_microbatch(result.valid_positions.detach())
                except Exception as e:error=f'{type(e).__name__}: {e}'
                self.check_errors(error)
            local_input+=int(result.nonpadding_input_tokens)
            local_valid+=int(result.valid_positions.item())
        counts=self.gather((local_input,local_valid))
        total_input=sum(x[0] for x in counts);total_valid=sum(x[1] for x in counts)
        self.check_errors(None if total_input==expected_global_input_tokens else 'Actual input count differs from frozen schedule')
        if self.cursor.global_input_tokens+total_input>self.input_token_cap:
            self.failed=True;raise RuntimeError('Actual token budget exceeded')
        lr_used=float(self.master.optimizer.param_groups[0]['lr'])
        error=None;grad_norm=None
        try:
            grad_norm=self.master.step(distributed=self.distributed)
            self.scheduler.step()
        except Exception as e:error=f'{type(e).__name__}: {e}'
        # A failed mutation cannot be retried or checkpointed as a valid update.
        self.check_errors(error)
        self.cursor.completed_optimizer_steps+=1;self.cursor.next_microbatch+=4
        self.cursor.global_input_tokens+=total_input;self.cursor.global_valid_positions+=total_valid
        assert self.master.completed_steps==self.cursor.completed_optimizer_steps
        return {**asdict(self.cursor),'global_input_tokens_this_update':total_input,
                'global_valid_positions_this_update':total_valid,'grad_norm_before_clip':grad_norm,
                'lr_used':lr_used,'lr_next':self.master.optimizer.param_groups[0]['lr']}

    def save_checkpoint(self,directory):
        """Commit only after all rank RNG/cursor files and FP32 state are hashed.

        No overwritten checkpoint, partial accumulation, or automatic recovery.
        This is a training-state checkpoint; native serving export is separate.
        """
        directory=Path(directory)
        error='Cannot checkpoint failed or partial window' if self.failed or self.master.microbatches else None
        self.check_errors(error)
        cursors=self.gather(asdict(self.cursor));schedulers=self.gather(self.scheduler.state_dict())
        agreements=self.gather(self.fingerprints)
        if any(x!=cursors[0] for x in cursors) or any(x!=schedulers[0] for x in schedulers) or any(x!=agreements[0] for x in agreements):
            self.failed=True;raise RuntimeError('Ranks disagree on committed checkpoint boundary')
        rng=None;error=None
        try:rng=capture_rng()
        except Exception as e:error=repr(e)
        self.check_errors(error)
        if self.rank==0:
            try:directory.mkdir(parents=True,exist_ok=False)
            except Exception as e:error=repr(e)
        self.check_errors(error)
        files=[];error=None
        try:
            files.append(save_file(directory/f'rank{self.rank}.pt',{'rank':self.rank,'rng':rng,'cursor':asdict(self.cursor)},tensor=True))
            if self.rank==0:
                files.append(save_file(directory/'master_optimizer.pt',self.master.state_dict(),tensor=True))
                files.append(save_file(directory/'metadata.json',{
                    'schema_version':1,'world_size':self.world,'cursor':asdict(self.cursor),
                    'scheduler':self.scheduler.state_dict(),'fingerprints':self.fingerprints,
                    'total_updates':self.total_updates,'input_token_cap':self.input_token_cap,
                    'checkpoint_type':'training_state_only_native_serving_export_pending'}))
        except Exception as e:error=repr(e)
        self.check_errors(error)
        all_files=[x for group in self.gather(files) for x in group];error=None
        if self.rank==0:
            try:
                for item in all_files:
                    p=directory/item['file']
                    if p.stat().st_size!=item['bytes'] or digest_file(p)!=item['sha256']:raise RuntimeError('Post-write verification failed')
                save_file(directory/'COMPLETE.json',{'schema_version':1,'files':all_files,'world_size':self.world})
                if os.name!='nt':
                    fd=os.open(directory,os.O_RDONLY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
            except Exception as e:error=repr(e)
        self.check_errors(error)
        return directory

    def restore_checkpoint(self,directory):
        directory=Path(directory);error=None;meta=None;rank_state=None;state=None
        try:
            if self.failed or self.master.microbatches or self.master.completed_steps or self.cursor.completed_optimizer_steps:
                raise ValueError('Restore requires fresh, empty engine')
            commit=json.loads((directory/'COMPLETE.json').read_text())
            expected={'metadata.json','master_optimizer.pt'}|{f'rank{r}.pt' for r in range(self.world)}
            if commit['world_size']!=self.world or {x['file'] for x in commit['files']}!=expected or len(commit['files'])!=len(expected):
                raise ValueError('Checkpoint file set/topology mismatch')
            # Rank zero verifies every payload; each rank also verifies what it loads.
            for item in commit['files']:
                if self.rank==0 or item['file'] in ('metadata.json','master_optimizer.pt',f'rank{self.rank}.pt'):
                    p=directory/item['file']
                    if p.stat().st_size!=item['bytes'] or digest_file(p)!=item['sha256']:raise ValueError('Checkpoint hash mismatch')
            meta=json.loads((directory/'metadata.json').read_text())
            if meta['schema_version']!=1 or meta['world_size']!=self.world:raise ValueError('Metadata schema/topology mismatch')
            if meta['fingerprints']!=self.fingerprints:raise ValueError('Source/data/schedule/weight fingerprints changed')
            if meta['total_updates']!=self.total_updates or meta['input_token_cap']!=self.input_token_cap:raise ValueError('Frozen budget changed')
            # These are exclusively our own hash-verified state artifacts.
            rank_state=torch.load(directory/f'rank{self.rank}.pt',map_location='cpu',weights_only=False)
            state=torch.load(directory/'master_optimizer.pt',map_location='cpu',weights_only=False)
            if rank_state['rank']!=self.rank or rank_state['cursor']!=meta['cursor']:raise ValueError('Rank cursor mismatch')
            if state['completed_steps']!=meta['cursor']['completed_optimizer_steps']:raise ValueError('Optimizer step mismatch')
            if meta['cursor']['next_microbatch']!=4*state['completed_steps']:raise ValueError('Sampler cursor mismatch')
            cursor=Cursor(**meta['cursor'])
            if any(type(v)!=int or v<0 for v in asdict(cursor).values()):raise ValueError('Invalid cursor values')
            if cursor.completed_optimizer_steps>self.total_updates or cursor.global_input_tokens>self.input_token_cap:raise ValueError('Checkpoint exceeds frozen budget')
            if meta['scheduler']['last_epoch']!=cursor.completed_optimizer_steps:raise ValueError('Scheduler cursor mismatch')
            if state['names']!=[n for n,_ in self.master.model] or state['expected_microbatches']!=4:raise ValueError('Master parameter names/window changed')
            for saved,master in zip(state['master'],self.master.master,strict=True):
                if saved.dtype!=torch.float32 or saved.shape!=master.shape or not torch.isfinite(saved).all().item():raise ValueError('Invalid FP32 master payload')
            groups=state['optimizer']['param_groups']
            if len(groups)!=1 or len(groups[0]['params'])!=len(self.master.master):raise ValueError('Optimizer parameter topology changed')
            ids=groups[0]['params']
            if len(set(ids))!=len(ids):raise ValueError('Duplicate optimizer parameter IDs')
            expected_states=set(ids) if cursor.completed_optimizer_steps else set()
            if set(state['optimizer']['state'])!=expected_states:raise ValueError('Optimizer state coverage mismatch')
            for ident,master in zip(ids,self.master.master,strict=True):
                if not cursor.completed_optimizer_steps:continue
                entry=state['optimizer']['state'][ident]
                if entry['step'].numel()!=1 or entry['step'].item()!=cursor.completed_optimizer_steps:raise ValueError('Per-parameter Adam step mismatch')
                for key in ('exp_avg','exp_avg_sq'):
                    tensor=entry[key]
                    if tensor.dtype!=torch.float32 or tensor.shape!=master.shape or not torch.isfinite(tensor).all().item():raise ValueError('Invalid FP32 Adam payload')
            validate_rng(rank_state['rng'])
        except Exception as e:error=repr(e)
        self.check_errors(error)
        error=None
        try:
            self.master.load_state_dict(state);self.scheduler.load_state_dict(meta['scheduler'])
            self.cursor=Cursor(**meta['cursor'])
            restore_rng(rank_state['rng'])
        except Exception as e:error=repr(e)
        self.check_errors(error)
        return asdict(self.cursor)
