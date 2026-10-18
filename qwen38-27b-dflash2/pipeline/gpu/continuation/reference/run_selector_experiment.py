# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""One locked 500-step paired experiment. No restart or automatic retuning."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import urllib.request

O = Path(__file__).resolve().parent
B = Path('/workspace')
W = B/'dflash2_work'
OLD = W/'public_reproduction/qwen38_27b_pb900kv2_ep4cont_20261004'
INITIAL = OLD/'run/epoch_snapshots/cont_ep1'
PY = str(W/'venv/bin/python')
TARGET = W/'models/Qwen3.8-27B'


def dump(p, data):
    p = Path(p); p.parent.mkdir(exist_ok=True, parents=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str)); temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(pid):
    p = Path('/proc')/str(pid)
    try:
        st = (p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid, 'starttime':st[19], 'state':st[0], 'ppid':int(st[1]),
                'pgid':int(st[2]), 'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:
        return None


def alive(expected):
    actual = identity(expected['pid'])
    return actual and actual['starttime']==expected['starttime'] and actual['state']!='Z'


def status(state, **kw):
    dump(O/'status.json', {'state':state, 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw})


def descendants(pid):
    found = {pid:identity(pid)}
    for _ in range(10):
        added = 0
        for p in Path('/proc').iterdir():
            if p.name.isdigit() and int(p.name) not in found:
                rec = identity(int(p.name))
                if rec and rec['ppid'] in found:
                    found[rec['pid']] = rec; added += 1
        if not added: break
    return [x for x in found.values() if x]


def cleanup_teachers(services):
    all_owned=[]
    for expected in services:
        if alive(expected):
            actual=identity(expected['pid'])
            assert actual['cmdline']==expected['cmdline']
            all_owned.extend(descendants(expected['pid']))
    dump(O/'teacher_cleanup_before.json',all_owned)
    # Only exact PID/starttime pairs captured under the verified service roots.
    for rec in all_owned:
        if alive(rec): os.kill(rec['pid'], signal.SIGTERM)
    for _ in range(30):
        if not any(alive(r) for r in all_owned): break
        time.sleep(1)
    remain=[r for r in all_owned if alive(r)]
    dump(O/'teacher_cleanup.json',{'remaining':remain,'observed':len(all_owned)})
    if remain: raise RuntimeError('Owned teacher cleanup incomplete; no forced broad kill')


def main():
    lock=(O/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (O/'status.json').exists(): raise RuntimeError('Refusing to overwrite existing experiment')
    assert json.loads((O/'preflight.json').read_text())['passed']
    assert shutil.disk_usage(O).free > 4*1024**3
    storage=Path('/models/draft_model_train/qwen38_27b_overlap_seed43_20261009_paired500_01')
    assert shutil.disk_usage(storage).free > 40*1024**3, 'Need durable NFS space for two native checkpoints'
    dump(O/'launch_identity.json',identity(os.getpid()))
    assert not any((O/arm/'trainer_identity.json').exists() for arm in ['hard_ce','overlap'])
    services=json.loads((O/'file_teacher_identities.json').read_text())
    for rec,port in zip(services[:4],range(8260,8264)):
        assert alive(rec)
        assert str(TARGET) in rec['cmdline'] and str(port) in rec['cmdline']
        assert urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=10).status==200
    assert alive(services[4]) and str(O/'teacher_dp4_router.py') in services[4]['cmdline']
    dump(O/'reused_teacher_identities.json',services)
    common=['--verifier-name-or-path',str(TARGET),'--from-pretrained',str(INITIAL),
            '--data-path',str(B/'prepared_pen0'),'--vllm-endpoint','http://127.0.0.1:8250/v1',
            '--log-freq','20','--speculator-type','dflash2','--block-size','8','--max-anchors','1024',
            '--mask-token-id','248070','--target-layer-ids','6','20','34','48','62',
            '--loss-fn','{"ce": 0.1, "tv": 0.9}','--dflash-decay-gamma','4.0',
            '--per-position-loss-weight','fixed-exp-decay','--selector-loss-alpha','0.1',
            '--optimizer','adamw','--lr','0.0003','--weight-decay','0',
            '--scheduler-type','cosine','--scheduler-warmup-ratio','0.02',
            '--scheduler-total-steps','500','--max-steps','500','--epochs','1',
            '--total-seq-len','8192','--train-data-ratio','0.99','--noise-std','0.05','--seed','43',
            '--hidden-states-dtype','bfloat16','--hidden-states-backend','file',
            '--hidden-states-path','/dev/shm/selector_overlap_seed43_20261009_01',
            '--num-workers','8','--prefetch-factor','2','--on-missing','generate','--on-generate','delete',
            '--request-timeout','600','--max-retries','3','--checkpoint-freq','1','--no-resume-from-checkpoint']
    protocol={'question':'Does replacing selector hard CE with 1-sum(min(p_full_on_K,q_K)) improve real T1 tau at fixed data, starting weights, alpha, and training budget?',
              'arms':['hard_ce','overlap'], 'training_candidate_policy':'Same native top16 plus injection of missing teacher argmax in both arms',
              'selector_stop_gradient':False,'teacher_temperature':1.,'teacher_training_filters':'No top-p/top-k/presence filters',
              'data':str(B/'prepared_pen0'),'initial_checkpoint':str(INITIAL),
              'initial_model_sha256':sha(INITIAL/'model.safetensors'),
              'training_seed':43,'replication_of':str(Path('/workspace/dflash2_work/public_reproduction/qwen38_27b_selector_overlap_20261008/paired_500_01')),'steps_per_arm':500,'world_size_per_arm':2,'fresh_optimizers':True,
              'common_args':common,'gpu_hard_ce':[4,5],'gpu_overlap':[6,7],
              'teacher_gpus':[0,1,2,3],'transport':'Shared tmpfs files, delete consumed handles, fail immediately on dropped records',
              'validation':'Same prefixes/anchors, baseline and final, fixed RNG restored around validation',
              'primary_endpoint':'Exploratory paired request mean tau on fixed 160 T1 prompts, C1, probabilistic draft, thinking+xhigh, T1/p.95/k20/presence0/max4096/spec7',
              'guardrail':'T0 GSM8K same prompts, greedy draft, presence0; exploratory',
              'limits':['500-step short adaptation is not proof about full-scale data scaling',
                        'Loss magnitude/gradient scale changes with objective; alpha stays 0.1',
                        'Teacher-forced unfiltered overlap is not real serving tau',
                        'Independent training seed43 replication; prompts previously evaluated; not held-out prompt confirmation'],
              'checkpoint_storage':str(storage),'selector_objective_detail':'overlap arm uses full vocabulary teacher log-normalizer; same injected K candidate set; teacher detached; all81 draft tensors trained; no selector freezing; no stop-gradient through unary logits or hidden states',
              'source_hashes':{str(p.relative_to(O)):sha(p) for p in O.rglob('*.py')},
              'data_hashes':{str(p):sha(p) for p in (B/'prepared_pen0').rglob('*') if p.is_file()}}
    dump(O/'PROTOCOL.json',protocol)
    status('launching')
    processes={}
    for arm,gpus,port in [('hard_ce','4,5',29708),('overlap','6,7',29709)]:
        root=O/arm;root.mkdir()
        destination=storage/arm/'checkpoints';destination.mkdir(parents=True,exist_ok=False)
        (root/'checkpoints').symlink_to(destination,target_is_directory=True)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpus,SELECTOR_OBJECTIVE=arm,EXPERIMENT_ARM_DIR=str(root),
                   PYTHONPATH=str(O/'source/src')+':'+str(O),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   TORCHINDUCTOR_CACHE_DIR=str(Path('/dev/shm/selector_overlap_seed43_compile_20261009')/arm/'inductor'),TRITON_CACHE_DIR=str(Path('/dev/shm/selector_overlap_seed43_compile_20261009')/arm/'triton'))
        command=[PY,'-m','torch.distributed.run','--nproc_per_node','2','--master_port',str(port),str(O/'run_selector_arm.py'),
                 *common,'--save-path',str(root/'checkpoints'),'--log-dir',str(root/'logs'),'--run-name','selector_'+arm]
        dump(root/'command.json',{'argv':command,'env':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','SELECTOR_OBJECTIVE','PYTHONPATH']}})
        log=(root/'trainer.log').open('wb')
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[arm]=p;dump(root/'trainer_identity.json',identity(p.pid))
    deadline=time.monotonic()+7200
    while any(p.poll() is None for p in processes.values()):
        assert time.monotonic()<deadline,'Two-hour fixed training deadline; inspect live owned workers'
        assert all(p.poll() in [None,0] for p in processes.values()),'Trainer failure; inspect owned resources, no retry'
        progress={}
        for arm in processes:
            progress[arm]=[json.loads(p.read_text()) for p in (O/arm/'audit').glob('progress_rank*.json')]
        status('training',progress=progress,free_gib=shutil.disk_usage(O).free/1024**3)
        time.sleep(15)
    exits={arm:p.returncode for arm,p in processes.items()}
    dump(O/'trainer_exits.json',exits)
    assert all(v==0 for v in exits.values()), exits
    checks={}
    for rank in range(2):
        arms={arm:json.loads((O/arm/f'audit/initial_rank{rank}.json').read_text()) for arm in processes}
        assert arms['hard_ce']['parameter_sha256']==arms['overlap']['parameter_sha256']
        records={arm:[json.loads(line) for line in (O/arm/f'audit/inputs_rank{rank}.jsonl').read_text().splitlines()] for arm in processes}
        assert records['hard_ce']==records['overlap'], f'Paired actual inputs/anchors differ on rank {rank}'
        for arm in processes:
            rows=records[arm]
            train=[r for r in rows if r['phase']=='train']
            assert len(train)==500 and [r['step'] for r in train]==list(range(500))
            base=[{k:v for k,v in r.items() if k!='phase'} for r in rows if r['phase']=='baseline']
            final=[{k:v for k,v in r.items() if k!='phase'} for r in rows if r['phase']=='final']
            assert base==final
            assert json.loads((O/arm/f'audit/completed_rank{rank}.json').read_text())['optimizer_steps']==500
        checks[str(rank)]={'paired_records':len(records['hard_ce']),'training_steps':500,'baseline_final_inputs_anchors_equal':True}
    assert sha(INITIAL/'model.safetensors')==protocol['initial_model_sha256']
    for p,h in protocol['data_hashes'].items():assert sha(p)==h
    result={'rank_checks':checks,'initial_unchanged':True,'data_unchanged':True,'arms':{}}
    for arm in processes:
        ck=O/arm/'checkpoints/0'
        assert json.loads((ck/'training_state.json').read_text())['global_step']==500
        result['arms'][arm]={'checkpoint':str(ck),'model_sha256':sha(ck/'model.safetensors'),
                             'baseline':json.loads((O/arm/'audit/baseline_rank0.json').read_text()),
                             'final':json.loads((O/arm/'audit/final_rank0.json').read_text())}
    dump(O/'training_verification.json',result)
    cleanup_teachers(services)
    status('ready_for_serving_evaluation',verification='training_verification.json')
    # Deliberately no retry or retuning. An audited evaluator is added separately.


if __name__=='__main__':
    try:main()
    except BaseException:
        dump(O/'failure.json',{'error':traceback.format_exc(),'time':time.time()})
        status('failed')
        raise
