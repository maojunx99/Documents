"""Paired native DFlash2 arm. Requires a separately sealed run specification.

This file does not start teachers or choose data. A smoke run uses a distinct
directory and never becomes a main-training initialization checkpoint.
"""
import argparse, datetime, hashlib, json, math, os, random, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from safetensors import safe_open
from datasets import load_from_disk
import experiment_audit as audit
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dflash2.metrics import SELECTOR_OBJECTIVE
from paired_training_engine import PairedTrainingEngine,ForwardNumerator,cosine_schedule,digest_file,capture_rng,restore_rng
from paired_native_data_v2 import ScheduledSharedDataset,StrictNativeCollate,LockedBatchSampler,NativeHiddenProducer,occurrence_seed
from bounded_feature_cache import PairedProgressGate,CacheQuota
from training_primitives import FP32MasterAdamW,EPS
from prepare_inventory import write,identity


def verify_files(files):
    for path,h in files.items():
        if digest_file(path)!=h:raise ValueError('Frozen input fingerprint changed: '+path)


def build_native(initial,target,device):
    config=DFlash2DraftModel.config_class.from_pretrained(initial,local_files_only=True)
    config.transformer_layer_config._attn_implementation='simple_flex_attention'
    if config.sample_from_anchor or config.block_size!=8 or config.selector_top_k!=16:
        raise ValueError('Unexpected model topology')
    model=DFlash2DraftModel.from_pretrained(initial,config=config,verifier=target,
                                           dtype=torch.bfloat16,local_files_only=True)
    model=model.to(device=device,dtype=torch.bfloat16)
    trainable=dict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    if len(trainable)!=81:raise ValueError('Expected all 81 draft trainable tensors')
    with safe_open(str(Path(initial)/'model.safetensors'),framework='pt',device='cpu') as f:
        if set(trainable)!=set(f.keys()):raise ValueError('Draft checkpoint tensor set differs')
        hashes={}
        for name,p in trainable.items():
            reference=f.get_tensor(name)
            if p.dtype!=torch.bfloat16 or not torch.equal(p.detach().cpu(),reference):
                raise ValueError('Loaded draft parameter differs: '+name)
            hashes[name]=audit.tensor_hash(p)
    frozen={n:audit.tensor_hash(p) for n,p in model.named_parameters() if not p.requires_grad}
    if not frozen or any(n in trainable for n in ('embed_tokens.weight','lm_head.weight','verifier_norm.weight')):
        raise ValueError('Verifier-owned parameters not frozen')
    kwargs,_=model.get_trainer_kwargs(loss_fn='{"ce":0.1,"tv":0.9}',loss_implementation='fused',
        dflash_decay_gamma=4.,max_anchors=1024,per_position_loss_weight='fixed-exp-decay',
        selector_loss_alpha=.1,selector_stop_gradient=False)
    return model,kwargs,hashes,frozen


def export_serving(model,directory,engine):
    """Native BF16 serving artifact alongside separately complete FP32 state."""
    error=None
    if engine.rank==0:
        try:
            directory=Path(directory);directory.mkdir(exist_ok=False)
            state={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
            if len(state)!=81 or any(x.dtype!=torch.bfloat16 for x in state.values()):raise ValueError('Invalid serving tensors')
            model.save_pretrained(directory,state_dict=state,safe_serialization=True,max_shard_size='20GB')
            with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as f:
                if set(f.keys())!=set(state):raise ValueError('Export tensor set differs')
                for n,p in state.items():
                    if not torch.equal(f.get_tensor(n),p):raise ValueError('Export tensor mismatch: '+n)
            write(directory/'EXPORT_VERIFICATION.json',dict(passed=True,tensors=81,dtype='bfloat16',
                model_sha256=digest_file(directory/'model.safetensors'),optimizer_steps=engine.cursor.completed_optimizer_steps,
                scope='BF16 inference model only; FP32 optimizer/master/RNG stored in sibling training_state'))
        except Exception as e:error=repr(e)
    engine.check_errors(error)


def main(spec_path,arm):
    spec=json.loads(Path(spec_path).read_text())
    deadline=time.monotonic()+spec['training_wall_seconds']
    if spec['schema_version']!=1 or spec['arms']!=['overlap','sampled_ce'] or arm not in spec['arms']:raise ValueError('Unknown protocol')
    if SELECTOR_OBJECTIVE!=arm or os.environ.get('SELECTOR_OBJECTIVE')!=arm:raise ValueError('Objective import/environment mismatch')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Locked DDP2 topology')
    root=Path(spec['run'])/arm;root.mkdir(parents=True,exist_ok=True);(root/'audit').mkdir(exist_ok=True)
    if Path(os.environ['EXPERIMENT_ARM_DIR'])!=root:raise ValueError('Audit directory mismatch')
    sentinel=root/f'rank{rank}.started';sentinel.open('x').close()
    write(root/f'audit/identity_rank{rank}.json',identity())
    verify_files(spec['fingerprinted_files'])
    if digest_file(spec['schedule'])!=spec['schedule_sha256']:raise ValueError('Schedule SHA mismatch')
    schedule=json.loads(Path(spec['schedule']).read_text())
    if schedule['world_size']!=2 or schedule['accumulation']!=4 or schedule['packed_budget']!=8192:raise ValueError('Schedule dimensions mismatch')
    if spec['mode']=='smoke':
        if schedule['total_updates']!=20 or not spec['isolated_smoke']:raise ValueError('Required smoke is exactly 20 updates')
    elif spec['mode']=='main':
        if not json.loads(Path(spec['gpu_smoke_verification']).read_text())['passed']:raise ValueError('GPU smoke not verified')
        if not json.loads(Path(spec['selection_lock']).read_text())['passed']:raise ValueError('Data not locked')
    else:raise ValueError('Unknown run mode')
    train_packs=schedule['train_packs'][str(rank)];val_packs=schedule['val_packs'][str(rank)]
    if len(train_packs)!=4*schedule['total_updates']:raise ValueError('Incomplete update windows')
    if len(schedule['global_input_tokens_per_update'])!=schedule['total_updates']:raise ValueError('Token schedule mismatch')
    torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15),device_id=device)
    random.seed(spec['training_seed']);np.random.seed(spec['training_seed']);torch.manual_seed(spec['training_seed']);torch.cuda.manual_seed_all(spec['training_seed'])
    native,call_kwargs,initial,frozen=build_native(spec['initial'],spec['target'],device)
    model=DDP(native,device_ids=[local],output_device=local,broadcast_buffers=False,find_unused_parameters=False)
    master=FP32MasterAdamW(native.named_parameters(),lr=1e-4,microbatches=4)
    if master.optimizer.state or any(x.dtype!=torch.float32 for x in master.master):raise ValueError('Nonempty or non-FP32 initialization')
    for (_,p),m in zip(master.model,master.master,strict=True):
        if not torch.equal(p.float(),m):raise ValueError('FP32 initialization copy mismatch')
    scheduler=cosine_schedule(master.optimizer,total_updates=schedule['total_updates'],warmup_updates=schedule['warmup_updates'])
    engine=PairedTrainingEngine(model,master,scheduler,total_updates=schedule['total_updates'],
        input_token_cap=spec['input_token_cap'],fingerprints={'schedule_sha256':spec['schedule_sha256'],**spec['fingerprinted_files']},distributed=True)
    gate=PairedProgressGate(Path(spec['shared_cache'])/'progress',arm,rank);gate.publish(0)
    records={phase:load_from_disk(spec[phase+'_dataset']) for phase in ['train','val']}
    producer=NativeHiddenProducer(spec['teacher_endpoint'],spec['target'],spec['transport_root'])
    def loader(phase):
        training=phase=='train';packs=train_packs if training else val_packs
        data=ScheduledSharedDataset(records['train' if training else 'val'],cache_root=spec['shared_cache'],arm=arm,
            producer=producer,seed=spec['training_seed'] if training else spec['validation_seed'])
        return DataLoader(data,batch_sampler=LockedBatchSampler(packs,phase=phase,rank=rank),
            num_workers=8,prefetch_factor=2,multiprocessing_context='spawn',persistent_workers=False,
            collate_fn=StrictNativeCollate(),pin_memory=True)
    write(root/f'audit/initial_rank{rank}.json',dict(parameter_sha256=initial,frozen_sha256=frozen,
        trainable_tensors=len(initial),trainable_parameters=sum(p.numel() for _,p in master.model),
        optimizer_states=len(master.optimizer.state),master_dtype='float32',model_dtype='bfloat16',
        workers_per_rank=8,accumulation=4,arm=arm,identity=identity(),spec_sha256=digest_file(spec_path),
        schedule_sha256=spec['schedule_sha256'],val_batches=len(val_packs),time=time.time()))
    metrics_log=(root/f'audit/metrics_rank{rank}.jsonl').open('x');features_log=(root/f'audit/features_rank{rank}.jsonl').open('x')
    state={'phase':'unset','micro':0};window=[]
    def forward(ddp,batch):
        if time.monotonic()>deadline:raise TimeoutError('Frozen arm wall limit including initialization/validation')
        a=batch.pop('_audit');phase=state['phase'];micro=state['micro']
        if any(x['phase']!=phase or x['microbatch']!=micro or x['rank']!=rank for x in a['samples']):raise ValueError('Occurrence mismatch')
        features_log.write(json.dumps(a)+'\n');features_log.flush()
        data={k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        audit.PHASE=phase;audit.STEP=micro;audit.VAL_INDEX=micro
        seed=occurrence_seed(spec['training_seed'] if phase=='train' else spec['validation_seed'],phase,rank,micro,-1)
        with torch.random.fork_rng(devices=[local]):
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            _,loss,metrics=ddp(**data,**call_kwargs)
        valid=metrics['valid_positions_sum'].detach()
        if not torch.isfinite(loss).item() or loss.ndim:raise FloatingPointError('Nonfinite native loss')
        scalars={k:float(v.detach().item()) for k,v in metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}
        if any(not math.isfinite(v) for v in scalars.values()):raise FloatingPointError('Nonfinite native metric')
        rec=dict(phase=phase,microbatch=micro,anchor_seed=seed,loss=float(loss.detach()),valid_positions=int(valid),metrics=scalars,time=time.time())
        metrics_log.write(json.dumps(rec)+'\n');metrics_log.flush();window.append(rec)
        state['micro']+=1
        return ForwardNumerator(loss*(valid+EPS),valid,a['nonpadding_input_tokens'])
    def validation(phase):
        rng=capture_rng();model.eval();state.update(phase=phase,micro=0);window.clear();totals=collections_counter()
        try:
            with torch.no_grad():
                for batch in loader(phase):
                    result=forward(model,batch)
                    for k,v in window[-1]['metrics'].items():totals[k]+=v
            all_totals=engine.gather(dict(totals));combined=collections_counter()
            for t in all_totals:
                for k,v in t.items():combined[k]+=v
            ratios={k[:-4]:v/combined[k[:-4]+'_total'] for k,v in combined.items() if k.endswith('_sum') and combined.get(k[:-4]+'_total',0)>0}
            write(root/f'audit/{phase}_rank{rank}.json',dict(completed_optimizer_steps=engine.cursor.completed_optimizer_steps,batches=state['micro'],summed=dict(combined),ratios=ratios,time=time.time()))
        finally:restore_rng(rng);model.train()
    validation('baseline');dist.barrier();state.update(phase='train',micro=0);iterator=iter(loader('train'))
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    for update in range(1,schedule['total_updates']+1):
        if time.monotonic()>deadline:raise TimeoutError('Frozen training wall limit')
        gate.wait_before_update(update);window.clear();batches=[next(iterator) for _ in range(4)]
        result=engine.update(batches,forward,expected_global_input_tokens=schedule['global_input_tokens_per_update'][update-1])
        gate.publish(update);result.update(time=time.time(),elapsed_s=time.monotonic()-started,
            mean_microbatch_loss=sum(x['loss'] for x in window)/4,peak_memory_bytes=torch.cuda.max_memory_allocated(device))
        if update==1 or update%10==0 or update==schedule['total_updates']:
            for st in master.optimizer.state.values():
                if st['exp_avg'].dtype!=torch.float32 or st['exp_avg_sq'].dtype!=torch.float32:raise ValueError('Adam moments not FP32')
            write(root/f'audit/progress_rank{rank}.json',result)
        if update in schedule['checkpoint_updates']:
            destination=Path(spec['checkpoint_root'])/arm/f'update_{update:06d}'
            if rank==0:destination.mkdir(parents=True,exist_ok=False)
            dist.barrier();engine.save_checkpoint(destination/'training_state');export_serving(native,destination/'serving',engine)
    try:next(iterator)
    except StopIteration:pass
    else:raise RuntimeError('Schedule has extra microbatches')
    del iterator;validation('final')
    for name,p in native.named_parameters():
        if not p.requires_grad and audit.tensor_hash(p)!=frozen[name]:raise ValueError('Frozen verifier parameter changed: '+name)
    write(root/f'audit/completed_rank{rank}.json',dict(completed_optimizer_steps=engine.cursor.completed_optimizer_steps,
        actual_input_tokens=engine.cursor.global_input_tokens,next_microbatch=engine.cursor.next_microbatch,
        final_parameter_sha256={n:audit.tensor_hash(p) for n,p in native.named_parameters() if p.requires_grad},
        frozen_parameters_unchanged=True,optimizer_moments_dtype='float32',time=time.time()))
    metrics_log.close();features_log.close();dist.barrier();dist.destroy_process_group()


def collections_counter():
    from collections import Counter
    return Counter()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--arm',required=True,choices=['overlap','sampled_ce']);a=p.parse_args()
    try:main(a.spec,a.arm)
    except BaseException as e:
        root=Path(os.environ['EXPERIMENT_ARM_DIR']);root.mkdir(parents=True,exist_ok=True)
        write(root/f'audit/failure_rank{os.environ.get("RANK","0")}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
