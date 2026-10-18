# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Frozen 697-update paired main training, fresh FP32 state from original start.
No smoke weights or mutable data. Post-training evaluation is separately gated.
"""
import argparse, collections, fcntl, hashlib, json, os, socket, subprocess, time, traceback, urllib.request
from pathlib import Path
from prepare_inventory import write
from run_additional_generation import ident,alive,cleanup,available_gpus,digest

W=Path('/workspace/dflash2_work')
REPO=Path('/workspace/speculators')
PY=str(W/'venv/bin/python');TARGET=W/'models/Qwen3.8-27B'
INITIAL=W/'public_reproduction/qwen38_27b_overlap_seed43_20261009/paired_500_01/overlap/checkpoints/0'
INITIAL_SHA='4cdb21b96156ac84abbf062cf93619e8f21daea0e18ea1ae6dce8ebaf52f7a8d'


def main(root,storage):
    child=root/'main_7gpu_01';child.mkdir(exist_ok=False)
    lock=(child/'runner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(child/'launch_identity.json',ident(os.getpid()))
    deadline=time.monotonic()+36000
    resource_deadline=time.monotonic()+3600
    smoke=json.loads((root/'gpu_smoke_verification.json').read_text());assert smoke['passed'] and smoke['native_CUDA_restore_four_ranks']
    assert json.loads((root/'SMOKE_LOCAL_SYNC_RECEIPT.json').read_text())['passed']
    def status(phase,**kw):
        rows=[x for v in kw.get('progress',{}).values() for x in v]
        started=(bool(rows) and min(x['completed_optimizer_steps'] for x in rows)>0) or (child/'completion_pending_audit.json').exists()
        write(child/'status.json',dict(phase=phase,time=time.time(),main_training_started=started,**kw))
    data=storage/'selection_v3_03';selection=json.loads((data/'SELECTION_LOCK.json').read_text());assert selection['passed']
    for n in ['native_arm_cpu_interface_preflight.json','boundary_guard_preflight.json','native_metrics_boundary_preflight.json','training_engine_preflight.json','paired_native_data_preflight.json','cache_bounds_preflight.json']:
        assert json.loads((root/n).read_text())['passed'],n
    assert digest(INITIAL/'model.safetensors')==INITIAL_SHA
    def wait_resources():
        while not {0,1,2,3,5,6,7}<=set(available_gpus()):
            if time.monotonic()>resource_deadline:raise TimeoutError('Selected7GPU resource wait1h exceeded')
            status('waiting_for_selected7_free',required=[0,1,2,3,5,6,7],available=available_gpus(),optimizer_steps=0)
            time.sleep(3)
    wait_resources()
    for port in [8250,8260,8261,8262,29720,29721]:
        with socket.socket() as sock:
            sock.settimeout(1);assert sock.connect_ex(('127.0.0.1',port))!=0,'Port busy: '+str(port)
    persistent=storage/'main_7gpu_01';persistent.mkdir(exist_ok=False)
    probe=persistent/'storage_probe.bin';payload=os.urandom(4096)
    with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    assert probe.read_bytes()==payload
    cache=Path('/dev/shm/qwen38_targeted_main_7gpu_20261009_01');cache.mkdir(exist_ok=False)
    write(cache/'owner.json',dict(run=str(child),identity=ident(os.getpid()),time=time.time()))
    schedule=json.loads((data/'native_schedule.json').read_text())
    assert schedule['total_updates']==697 and schedule['warmup_updates']==34
    assert schedule['checkpoint_updates']==[175,349,523,697]
    assert digest(data/'native_schedule.json')==selection['schedule_sha256']
    write(child/'native_schedule.json',schedule)
    protected={str(p):digest(p) for base in [TARGET,INITIAL,root/'source_integration_v2/src',root/'reference'] for p in base.rglob('*') if p.is_file() and p.suffix not in ('.pyc',) and '__pycache__' not in p.parts}
    for n in ['train_targeted_arm_v3.py','paired_training_engine.py','training_primitives.py','paired_native_data_v2.py','bounded_feature_cache.py','shared_feature_cache.py','training_boundary_guard.py','targeted_teacher_router_7gpu.py', 'run_targeted_main_7gpu_v2.py']:
        protected[str(root/n)]=digest(root/n)
    for n in ['train_manifest.jsonl','val_manifest.jsonl','SELECTION_LOCK.json']:
        protected[str(data/n)]=digest(data/n)
    for base in [data/'prepared_train',data/'prepared_val']:
        for p in base.rglob('*'):
            if p.is_file():protected[str(p)]=digest(p)
    for name in ['PLAN.json','PLAN_DATA_SCALE_REVISION_v3.json','gpu_smoke_verification.json']:
        protected[str(root/name)]=digest(root/name)
    write(child/'fingerprints_before.json',protected)
    spec=dict(schema_version=1,mode='main',isolated_smoke=False,gpu_smoke_verification=str(root/'gpu_smoke_verification.json'),arms=['overlap','sampled_ce'],
        run=str(child),initial=str(INITIAL),target=str(TARGET),training_seed=2026100902,validation_seed=2026100903,
        train_dataset=str(data/'prepared_train'),val_dataset=str(data/'prepared_val'),schedule=str(child/'native_schedule.json'),
        schedule_sha256=digest(child/'native_schedule.json'),fingerprinted_files=protected,
        shared_cache=str(cache/'shared'),transport_root=str(cache/'transport'),teacher_endpoint='http://127.0.0.1:8250/v1',
        checkpoint_root=str(persistent/'checkpoints'),input_token_cap=schedule['actual_input_tokens'],training_wall_seconds=28800,
        selection_lock=str(data/'SELECTION_LOCK.json'),main_training_started=False,resource_topology=dict(teachers=[0,1,2],overlap=[3,5],sampled_ce=[6,7],excluded=[4]))
    write(child/'RUN_SPEC.json',spec)
    protocol=dict(mode='main',updates_per_arm=697,accumulation=4,checkpoint_updates=schedule['checkpoint_updates'],
        input_tokens_per_arm=schedule['actual_input_tokens'],wall_training_seconds=28800,
        warmup_updates=34,main_init='Original current overlap checkpoint plus empty FP32 optimizer; no smoke reuse',
        evaluation=json.loads((root/'PLAN.json').read_text())['evaluation'],
        decision_rules=json.loads((root/'PLAN.json').read_text())['decision_rules'],
        resource_topology=spec['resource_topology'])
    write(child/'PROTOCOL.json',protocol)
    services=[];server_processes=[];trainers={};root_env=os.environ.copy()
    root_env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    try:
        wait_resources()
        for gpu in range(3):
            d=child/'teachers'/str(gpu);d.mkdir(parents=True)
            cmd=[PY,str(REPO/'scripts/launch_vllm.py'),'train',str(TARGET),
                '--hidden-states-backend','file','--hidden-states-path',str(cache/'transport'),
                '--target-layer-ids','6','20','34','48','62','--include-last-layer','--provenance-dir',str(d/'provenance'),
                '--tensor-parallel-size','1','--max-model-len','16384','--api-server-count','1','--mm-processor-cache-gb','0','--port',str(8260+gpu)]
            env={**root_env,'CUDA_VISIBLE_DEVICES':str(gpu),'VLLM_CACHE_ROOT':str(cache/'teacher_compile'/str(gpu))}
            write(d/'command.json',dict(argv=cmd,env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','VLLM_CACHE_ROOT']}))
            with (d/'server.log').open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
            server_processes.append(p);services.append(ident(p.pid));write(d/'identity.json',services[-1]);write(child/'teacher_identities_initial.json',services)
        cmd=[PY,str(root/'targeted_teacher_router_7gpu.py')]
        with (child/'router.log').open('xb') as f:p=subprocess.Popen(cmd,env={**root_env,'TARGETED_RUNTIME_DIR':str(child)},stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
        server_processes.append(p);services.append(ident(p.pid));write(child/'teacher_identities_initial.json',services)
        startup=min(deadline,time.monotonic()+1200)
        while True:
            assert all(p.poll() is None for p in server_processes),'Teacher exited during startup'
            ready=[]
            for port in range(8260,8263):
                try:ready.append(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).status==200)
                except Exception:ready.append(False)
            status('starting_smoke_teachers',ready=ready)
            if all(ready):break
            if time.monotonic()>startup:raise TimeoutError('Teacher startup limit')
            time.sleep(5)
        updated=[ident(r['pid']) for r in services]
        assert all(a['starttime']==b['starttime'] for a,b in zip(services,updated));services=updated
        write(child/'teacher_identities.json',services)
        for gpu,rec in enumerate(services[:3]):write(child/'teachers'/str(gpu)/'identity.json',rec)
        for arm,gpus,port in [('overlap','3,5',29720),('sampled_ce','6,7',29721)]:
            d=child/arm;d.mkdir();env={**root_env,'CUDA_VISIBLE_DEVICES':gpus,'SELECTOR_OBJECTIVE':arm,'EXPERIMENT_ARM_DIR':str(d),
                'PYTHONPATH':':'.join(map(str,[root/'source_integration_v2/src',root/'reference',root])),
                'TORCHINDUCTOR_CACHE_DIR':str(cache/'compiler'/arm/'inductor'),'TRITON_CACHE_DIR':str(cache/'compiler'/arm/'triton')}
            cmd=[PY,'-m','torch.distributed.run','--nproc_per_node','2','--master_port',str(port),str(root/'train_targeted_arm_v3.py'),'--spec',str(child/'RUN_SPEC.json'),'--arm',arm]
            write(d/'command.json',dict(argv=cmd,env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','SELECTOR_OBJECTIVE','EXPERIMENT_ARM_DIR','PYTHONPATH']}))
            with (d/'trainer.log').open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
            trainers[arm]=p;write(d/'trainer_identity.json',ident(p.pid))
            (d/'train_command.txt').write_text(' '.join(cmd)+'\n')
        while any(p.poll() is None for p in trainers.values()):
            if time.monotonic()>deadline:raise TimeoutError('Whole main10h supervisor cap')
            assert all(p.poll() in (None,0) for p in trainers.values()),'Main trainer failed'
            assert all(p.poll() is None for p in server_processes),'Teacher/router failure'
            progress={arm:[json.loads(p.read_text()) for p in (child/arm/'audit').glob('progress_rank*.json')] for arm in trainers}
            status('main_training_or_baseline',progress=progress);time.sleep(10)
        write(child/'trainer_exits.json',{k:p.returncode for k,p in trainers.items()})
        assert all(p.returncode==0 for p in trainers.values())
        # Check pair consistency before declaring that this smoke even completed.
        initials=[]
        for arm in trainers:
            for rank in range(2):
                d=child/arm/'audit';z=json.loads((d/f'completed_rank{rank}.json').read_text())
                assert z['completed_optimizer_steps']==697 and z['actual_input_tokens']==schedule['actual_input_tokens']
                initials.append(json.loads((d/f'initial_rank{rank}.json').read_text())['parameter_sha256'])
        assert all(x==initials[0] for x in initials)
        for rank in range(2):
            for name in ['inputs','features']:
                a=[json.loads(l) for l in (child/'overlap'/'audit'/f'{name}_rank{rank}.jsonl').read_text().splitlines()]
                b=[json.loads(l) for l in (child/'sampled_ce'/'audit'/f'{name}_rank{rank}.jsonl').read_text().splitlines()]
                assert a==b and len(a)==4*697+2*len(schedule['val_packs'][str(rank)]),(name,rank,len(a),len(b))
        status('main_updates_completed_pending_independent_audit')
    except BaseException:
        # Preserve live trainers for exact diagnosis; do not strand their teacher
        # underneath them by eagerly clearing the transport.
        write(child/'failure.json',dict(error=traceback.format_exc(),time=time.time(),live_trainers={k:p.pid for k,p in trainers.items() if p.poll() is None},identity=ident(os.getpid())))
        raise
    else:
        cleanup(child,services)
        after={p:digest(p) for p in protected};assert after==protected
        write(child/'fingerprints_after.json',after)
        write(child/'completion_pending_audit.json',dict(updates_per_arm=697,owned_teachers_exited=True,source_target_initial_unchanged=True,
            main_training_started=False,next='Independent full audit, FP32 state/CUDA restore check and owned cache audit; no main auto-launch',time=time.time()))
        status('main_completed_pending_independent_audit_and_serving')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--storage',type=Path,required=True);a=p.parse_args()
    try:main(a.run,a.storage)
    except BaseException as e:
        child=a.run/'main_7gpu_01';child.mkdir(exist_ok=True)
        if not (child/'failure.json').exists():write(child/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=ident(os.getpid())))
        raise
