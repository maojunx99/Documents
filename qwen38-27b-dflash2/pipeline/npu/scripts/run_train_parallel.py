"""Audited FSDP/HSDP with explicit replica and shard dimensions; native losses/data unchanged."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
import speculators.train.trainer as native
from replica_audit import audit_replicas

options=argparse.ArgumentParser(add_help=False)
options.add_argument('--parallel-replicas',type=int,required=True)
options.add_argument('--parallel-shard-size',type=int,required=True)
options.add_argument('--parallel-stop-after-epoch',type=int)
parallel,remaining=options.parse_known_args()
X,Y=parallel.parallel_replicas,parallel.parallel_shard_size
RANK=int(os.environ['RANK'])
AUDIT=Path(os.environ['TRAIN_AUDIT']);AUDIT.mkdir(parents=True,exist_ok=True)
Trainer=native.Trainer
import speculators.train.data as native_data
original_get_raw_data=native_data.ArrowDataset._get_raw_data


def strict_get_raw_data(self,index):
    sample=original_get_raw_data(self,index)
    if sample is None:
        save(f'data_failure_pid{os.getpid()}',{'index':int(index),'rank':RANK,
             'reason':'Missing or invalid teacher hidden states; refuse silent CollateFn sample removal'})
        raise RuntimeError(f'Missing or invalid teacher sample {index}; refusing a dropped record')
    return sample


native_data.ArrowDataset._get_raw_data=strict_get_raw_data


def save(name,value):
    path=AUDIT/f'{name}_rank{RANK}.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def local(t):return t.to_local() if isinstance(t,DTensor) else t


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def apply_parallel(model,param_dtype=torch.bfloat16):
    assert X>=1 and Y>=1 and dist.get_world_size()==X*Y
    mesh=init_device_mesh('npu',(X,Y) if X>1 else (Y,),mesh_dim_names=('replicate','shard') if X>1 else ('shard',))
    policy=MixedPrecisionPolicy(param_dtype=param_dtype,reduce_dtype=torch.float32)
    for layer in model.layers:fully_shard(layer,mesh=mesh,mp_policy=policy)
    fully_shard(model,mesh=mesh,mp_policy=policy)
    model._training_mesh=mesh
    return model


native.apply_fully_sharded=apply_parallel
original_model=Trainer.setup_model


def setup_model(self):
    from speculators.models.utils import resolve_verifier_norm_class, uses_gemma_style_final_norm
    expected_norm=resolve_verifier_norm_class(self.model.config)
    assert isinstance(self.model.verifier_norm,expected_norm),'Verifier norm must match the actual target gain convention'
    self.model.supports_gradient_checkpointing=True
    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    self._initial_hashes={n:{'shape':list(p.shape),'dtype':str(p.dtype),'sha256':digest(p)} for n,p in self.model.named_parameters() if p.requires_grad}
    save('before_shard',{'trainable':self._initial_hashes,'random_initialization':True,
                        'verifier_norm_class':type(self.model.verifier_norm).__name__,
                        'verifier_norm_gain':'1+w' if uses_gemma_style_final_norm(self.model.config) else 'w'})
    original_model(self)


original_optimizer=Trainer.setup_optimizer


def setup_optimizer(self):
    original_optimizer(self)
    params={n:p for n,p in self.model.named_parameters() if p.requires_grad}
    full={n:{'shape':list(p.shape),'dtype':str(p.dtype),'sha256':digest(p.full_tensor())} for n,p in params.items()}
    assert full==self._initial_hashes
    assert params and all(isinstance(p,DTensor) and p.dtype==torch.float32 for p in params.values())
    assert all(len(o.state)==0 for o in self.optimizers)
    self._params=params;self._sample_name=next(iter(params))
    self._sample=local(params[self._sample_name]).detach().flatten()[:4096].float().cpu().clone()
    self._audit_started=time.time();self._audit_loss=None;self._audit_epoch=0
    mesh=self.model._training_mesh
    expected='(Replicate(), Shard(dim=0))' if X>1 else '(Shard(dim=0),)'
    assert all(str(p.placements)==expected for p in params.values())
    from speculators.train.distributed import get_dp_size
    assert get_dp_size()==X*Y
    save('initial',{'world_size':dist.get_world_size(),'replicas':X,'shard_size':Y,'mesh':mesh.mesh.tolist(),
         'shard_group':dist.get_process_group_ranks(mesh.get_group('shard')),
         'replica_group':dist.get_process_group_ranks(mesh.get_group('replicate')) if X>1 else [RANK],
         'train_steps':len(self.train_loader),'val_steps':len(self.val_loader),'trainable':full,
         'trainable_parameters_global':sum(p.numel() for p in params.values()),
         'trainable_parameters_local':sum(local(p).numel() for p in params.values()),
         'optimizer_initial_states':[len(o.state) for o in self.optimizers],
         'scheduler_total_steps':self.config.scheduler_total_steps,
         'scheduler_warmup_steps':self.config.scheduler_warmup_steps,
         'planned_epochs':self.config.num_epochs,'stop_after_epoch':parallel.parallel_stop_after_epoch,
         'placements':{n:str(p.placements) for n,p in params.items()},'sampler_dp_size':get_dp_size()})
    torch.npu.reset_peak_memory_stats()
    def output_hook(module,args,output):
        self._audit_loss=float(output[1].detach().float().item());assert math.isfinite(self._audit_loss)
    def input_hook(module,args,kwargs):
        error=kwargs.get('error_records',0)
        if isinstance(error,torch.Tensor):error=error.item()
        assert not error,'Dropped training/validation records'
        row={'phase':'train' if module.training else 'validation','epoch':self._audit_epoch,'global_step':self.global_step,
             'input_sha256':digest(kwargs['input_ids']),'mask_sha256':digest(kwargs['loss_mask']),
             'input_shape':list(kwargs['input_ids'].shape)}
        with (AUDIT/f'inputs_rank{RANK}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    self.model.register_forward_hook(output_hook);self.model.register_forward_pre_hook(input_hook,with_kwargs=True)


original_step=Trainer._optimizers_step


def optimizer_step(self):
    n=self.global_step+1
    grads=[torch.isfinite(local(p.grad)).all() for p in self._params.values() if p.grad is not None]
    assert grads and torch.stack(grads).all().item(),'Nonfinite gradient'
    if X>1 and n<=3:
        sample=digest(local(self._params[self._sample_name].grad).flatten()[:4096])
        gathered=[None]*(X*Y);dist.all_gather_object(gathered,sample)
        assert all(gathered[r]==gathered[RANK%Y] for r in range(RANK%Y,X*Y,Y))
        save(f'gradient_replicas_step{n}',{'passed':True,'all_rank_hashes':gathered})
    lr=self.optimizers[0].param_groups[0]['lr'];original_step(self)
    counters=[float(s['step'].item()) for o in self.optimizers for s in o.state.values() if 'step' in s]
    assert counters and min(counters)==max(counters)==n
    changed=bool((local(self._params[self._sample_name]).detach().flatten()[:4096].float().cpu()!=self._sample).any())
    row={'rank':RANK,'epoch':self._audit_epoch,'completed_optimizer_steps':n,'loss':self._audit_loss,'lr_used':lr,
         'optimizer_states':len(counters),'optimizer_step_min':min(counters),'optimizer_step_max':max(counters),
         'sample_weight_changed_since_initialization':changed,'elapsed_seconds':time.time()-self._audit_started,
         'npu_peak_allocated_bytes':torch.npu.max_memory_allocated(),'npu_peak_reserved_bytes':torch.npu.max_memory_reserved()}
    with (AUDIT/f'updates_rank{RANK}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    save('progress',row)
    if n in (1,3,100) or n%1000==0:
        assert audit_replicas(self._params,self.optimizers,self.model._training_mesh,
                              X,Y,save,f'step{n}'),'Replica difference; named payload saved before failure'


def run_training(self):
    epochs=[]
    for epoch in range(self.current_epoch,self.config.num_epochs):
        if parallel.parallel_stop_after_epoch is not None and epoch>parallel.parallel_stop_after_epoch:break
        if self.config.max_steps and self.global_step>=self.config.max_steps:break
        self._audit_epoch=epoch
        self.train_epoch(epoch);dist.barrier()
        assert audit_replicas(self._params,self.optimizers,self.model._training_mesh,
                              X,Y,save,f'epoch{epoch}_before_save'),'Replica difference before checkpoint'
        self.maybe_save_checkpoint(epoch);dist.barrier()
        assert audit_replicas(self._params,self.optimizers,self.model._training_mesh,
                              X,Y,save,f'epoch{epoch}_after_save'),'Replica difference after checkpoint'
        metrics=self.val_epoch(epoch)
        assert metrics and all(math.isfinite(v) for v in metrics.values())
        save(f'validation_epoch{epoch}',{'global_step':self.global_step,'metrics':metrics,'batches':len(self.val_loader)})
        dist.barrier();self.maybe_update_best(epoch,metrics);dist.barrier()
        assert audit_replicas(self._params,self.optimizers,self.model._training_mesh,
                              X,Y,save,f'epoch{epoch}_after_validation'),'Replica difference after validation'
        epochs.append(epoch)
    assert epochs and self.global_step>0
    last=epochs[-1]
    payload_path=f'replica_epoch{last}_after_validation_rank{RANK}.json'
    named=json.loads((AUDIT/payload_path).read_text())['all_rank_payloads'][RANK]['tensors']
    save('replica_state',{'passed':True,'replica_ranks':list(range(RANK%Y,X*Y,Y)),
                          'hashes':named,
                          'named_payload':payload_path})
    full=get_model_state_dict(self.model,options=StateDictOptions(full_state_dict=True,cpu_offload=True))
    if RANK==0:
        from safetensors.torch import load_file
        checkpoint=Path(self.config.save_path)/str(last)
        serialized=load_file(str(checkpoint/'model.safetensors'),device='cpu')
        assert all(digest(full[n].to(torch.bfloat16))==digest(serialized[n]) for n in self._params)
        expected={n:digest(serialized[n].float()) for n in self._params}
        saved_optimizer=torch.load(checkpoint/'optimizer_state_dict.pt',map_location='cpu',weights_only=False)
        assert len(saved_optimizer['state'])==len(self._params)
        del serialized,saved_optimizer
    del full;dist.barrier()
    self.checkpointer.previous_epoch=last
    self.checkpointer.load_model_state_dict(self.model);self.checkpointer.load_optimizer_state_dict(self.model,self.optimizers)
    hashes={n:digest(p.full_tensor()) for n,p in self._params.items()}
    save('reloaded_hashes',hashes);dist.barrier()
    if RANK==0:
        assert all(json.loads((AUDIT/f'reloaded_hashes_rank{i}.json').read_text())==expected for i in range(X*Y))
        save('checkpoint_reload',{'passed':True,'epoch':last,'model_load_executed_all_ranks':True,'optimizer_load_executed_all_ranks':True,
             'trainable_tensors':len(self._params),'all_ranks_match_serialized_bf16_expanded_fp32':True,
             'limitation':'Native BF16 model/moments and floating step; no RNG snapshot. Not lossless FP32 resume.'})
    dist.barrier()
    save('final',{'passed':True,'completed_optimizer_steps':self.global_step,'completed_epochs':epochs,'max_steps':self.config.max_steps,
                  'planned_epochs':self.config.num_epochs,'stop_after_epoch':parallel.parallel_stop_after_epoch})


Trainer.setup_model=setup_model;Trainer.setup_optimizer=setup_optimizer;Trainer._optimizers_step=optimizer_step;Trainer.run_training=run_training
try:
    script=Path(os.environ['NATIVE_SOURCE'])/'scripts/train.py'
    sys.argv=[str(script),*remaining];runpy.run_path(str(script),run_name='__main__')
    save('exit',{'exit_code':0,'utc_epoch':time.time()})
except BaseException:
    save('failure',{'traceback':traceback.format_exc(),'utc_epoch':time.time()});raise
