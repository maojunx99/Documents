"""Fresh-only NPU end-to-end27B workflow; never retries training or overwrites runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'common'))
from managed import OwnedProcesses, assert_free, identity, source_environment, write
from npu.recipe import parser, validate, command_plan


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def health(port,process,timeout=1200):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError(f'Server exited before readiness on port{port}')
        try:
            if urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).status==200:return
        except OSError:pass
        time.sleep(2)
    raise TimeoutError(f'Server readiness on port{port}')


def protected_files(source):
    paths=list(source.rglob('*.py'))
    paths+=list((ROOT/'regeneration/source').rglob('*.py'))
    paths+=list(BASE.glob('*.py'))+list((BASE/'scripts').glob('*.py'))
    paths+=list((ROOT/'common').glob('*.py'))+list((ROOT/'regeneration').glob('*.py'))
    paths+=[ROOT/'gpu/teacher_router.py']
    return {str(p):sha(p) for p in paths if '__pycache__' not in p.parts}


def main():
    a=parser().parse_args();resolved=validate(a);commands=command_plan(a,resolved)
    if a.stage != 'all':
        from npu.stages import execute
        return execute(a, resolved, commands)
    serial={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(a).items()}
    if a.dry_run:
        print(json.dumps({'config':serial,'rank_mesh':resolved['mesh'],'commands':commands,
                          'stage_order':['preflight','regeneration(if requested)','stop generation servers','prepare',
                                         'live teachers','x*FSDPy training','validation/save/reload','owned cleanup','completion audit']},indent=2))
        return
    if not a.model.is_dir():raise ValueError('A complete local target model directory is required')
    if a.input and any(not p.is_file() for p in a.input):raise ValueError('A selected raw prompt input is missing')
    if a.regenerated_dir and not a.regenerated_dir.is_dir():raise ValueError('regenerated-dir is missing')
    all_cards=resolved['train']+[d for g in resolved['teachers'] for d in g]
    assert_free('npu',all_cards)
    if a.input:assert_free('npu',[d for g in resolved['regeneration'] for d in g])
    a.run.mkdir(parents=True,exist_ok=False)
    run=a.run;source=BASE/'source';owners=[];teacher_processes=[];completed=False
    cache=a.hidden_cache.resolve() if a.hidden_cache else run/'hidden_states'
    if cache.exists():raise ValueError('hidden-cache must be a new directory, never an existing shared cache')
    if cache==run or run.is_relative_to(cache):raise ValueError('hidden-cache cannot contain the run directory')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    env=source_environment(source,{'TORCHDYNAMO_DISABLE':'1','TASK_QUEUE_ENABLE':'1','OMP_NUM_THREADS':'2','HCCL_EXEC_TIMEOUT':'600',
                                  'HCCL_CONNECT_TIMEOUT':'180','TRAIN_AUDIT':str(run/'audit'),'NATIVE_SOURCE':str(source)})
    cann=Path(env.get('ASCEND_HOME_PATH','/usr/local/Ascend/cann-9.1.0'))/'python/site-packages'
    env['PYTHONPATH']=os.pathsep.join([str(BASE/'scripts'),str(cann),env.get('PYTHONPATH','')])
    train_env=dict(env,ASCEND_RT_VISIBLE_DEVICES=a.train_devices)
    protected=protected_files(source)
    target={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                    'sha256':sha(p) if p.stat().st_size<1024*1024 else None}
            for p in a.model.iterdir() if p.is_file()}
    write(run/'PIPELINE_CONFIG.json',serial);write(run/'COMMAND_PLAN.json',commands)
    write(run/'launch_identity.json',identity(os.getpid()));write(run/'protected_source.json',protected)
    write(run/'target_manifest.json',target);write(run/'draft_config.json',resolved['draft'])
    # Every invocation records the actual target/tokenizer. Similar model names do not authorize reusing old answers.
    write(run/'MODEL_COMPATIBILITY.json',{'target':str(a.model),'hidden_size':resolved['draft']['hidden_size'],
          'vocab_size':resolved['draft']['vocab_size'],'feature_layers':a.target_layer_ids,'mask_token_id':a.mask_token_id,
          'target_config_sha256':sha(a.model/'config.json'),'draft_config_sha256':sha(a.draft_config)})
    def stage(name,**kwargs):write(run/'status.json',{'stage':name,'utc_epoch':time.time(),**kwargs})
    def new_owner(folder):
        folder.mkdir(parents=True,exist_ok=False);owner=OwnedProcesses(folder);owners.append(owner);return owner
    stage_owner=new_owner(run/'stage_processes')
    def checked(name,argv,current_env=env):
        p=stage_owner.start(name,argv,current_env,cwd=ROOT)
        deadline=time.monotonic()+a.stage_timeout
        while p.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError(name+' exceeded stage-timeout')
            if any(t.poll() is not None for t in teacher_processes):raise RuntimeError('A live teacher exited')
            time.sleep(2)
        if p.returncode:raise RuntimeError(name+' failed; inspect its log')
        return p
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
        vocab=tokenizer.get_vocab()
        if not vocab or max(vocab.values())>=resolved['target_text_config']['vocab_size']:
            raise ValueError('Tokenizer IDs exceed the target embedding vocabulary')
        mask_token=tokenizer.convert_ids_to_tokens(a.mask_token_id)
        if mask_token is None or tokenizer.convert_tokens_to_ids(mask_token)!=a.mask_token_id:
            raise ValueError('mask-token-id cannot round trip through this target tokenizer')
        token_probe=tokenizer.apply_chat_template([{'role':'user','content':'Compatibility check.'}],tokenize=True,add_generation_prompt=True)
        if hasattr(token_probe,'keys'):token_probe=token_probe['input_ids']
        if not token_probe or any(not 0<=i<resolved['target_text_config']['vocab_size'] for i in token_probe):
            raise ValueError('Target chat template produced invalid token IDs')
        compatibility=json.loads((run/'MODEL_COMPATIBILITY.json').read_text())
        compatibility.update(tokenizer_class=type(tokenizer).__name__,tokenizer_entries=len(vocab),
                             mask_token=mask_token,mask_token_round_trip=True,chat_template_probe_passed=True,
                             policy='Uses the selected target tokenizer and fresh target responses, not a tokenizer from old weights.')
        write(run/'MODEL_COMPATIBILITY.json',compatibility)
        del tokenizer,vocab
        for port in [a.router_port]+list(range(a.teacher_port,a.teacher_port+len(resolved['teachers']))):
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        if shutil.disk_usage(run).free<25*2**30:raise RuntimeError('Need at least25GiB free for model/optimizer checkpoints and staging')
        stage('preflight')
        (run/'preflight').mkdir()
        for name,script in [('kernels','preflight_npu.py'),('checkpointing','preflight_checkpointing.py'),('cpu_checkpoint','preflight_cpu_checkpoint.py')]:
            extra={'ASCEND_RT_VISIBLE_DEVICES':str(resolved['train'][0]),'AUDIT_OUT':str(run/f'preflight/{name}.json'),
                   'PREFLIGHT_ROOT':str(run/'preflight/cpu_checkpoint_artifacts')}
            checked('preflight_'+name,[a.python,str(BASE/'scripts'/script)],dict(env,**extra))
            receipt=json.loads((run/f'preflight/{name}.json').read_text())
            if name=='checkpointing':assert receipt['cpu']['passed'] and receipt['npu']['passed']
            else:assert receipt['passed']
        checked('parallel_preflight',commands['parallel_preflight'],train_env)
        for rank in range(a.replicas*a.shard_size):
            assert json.loads((run/f'preflight/parallel/passed_rank{rank}.json').read_text())['passed']
        if a.input:
            stage('regenerating_answers')
            generation_owner=new_owner(run/'generation_stage')
            generation=generation_owner.start('servers',commands['generation_server'],env,cwd=ROOT)
            # serve.py writes endpoints only after all of its servers have passed health checks.
            deadline=time.monotonic()+1200*len(resolved['regeneration'])
            while not (run/'generation_servers/endpoints.json').exists():
                if generation.poll() is not None:raise RuntimeError('Generation supervisor failed')
                if time.monotonic()>deadline:raise TimeoutError('Generation services readiness')
                time.sleep(2)
            checked('regeneration',commands['regeneration'])
            assert json.loads((run/'regenerated/COMPLETION.json').read_text())['passed']
            generation_owner.close()
            generation_cleanup=json.loads((run/'generation_servers/cleanup.json').read_text())
            assert all(not v['remaining'] for v in generation_cleanup)
            stage('generation_complete_services_released')
            data=run/'regenerated/shards'
        else:
            data=a.regenerated_dir.resolve()
            # A directory produced by regenerate.py must name exactly this target path.
            metadata=data.parent/'REGENERATION_CONFIG.json'
            if not metadata.exists():raise ValueError('Reused answers require their REGENERATION_CONFIG.json target manifest')
            recorded=json.loads(metadata.read_text())
            if Path(recorded['model']).resolve()!=a.model:raise ValueError('Reused answers name a different target model')
            if not recorded.get('target_model_files'):raise ValueError('Reused answers lack target_model_files identity; regenerate with this entry or supply an audited identity manifest')
            for name,item in recorded['target_model_files'].items():
                f=a.model/name
                if not f.is_file() or f.stat().st_size!=item['bytes'] or f.stat().st_mtime_ns!=item['mtime_ns']:
                    raise ValueError('Target files changed since these answers were generated: '+name)
                if item['sha256'] and sha(f)!=item['sha256']:raise ValueError('Target hash changed: '+name)
            write(run/'answer_reuse.json',{'manifest':str(metadata),'sha256':sha(metadata),'model':recorded['model']})
        sys.path.insert(0,str(ROOT/'regeneration'))
        from verify_outputs import verify
        verified=verify(data);write(run/'regenerated_data_verification.json',verified)
        source_rows=verified['rows']
        if min(int(source_rows*a.train_ratio),source_rows-int(source_rows*a.train_ratio))<a.replicas*a.shard_size:
            raise ValueError('Too few train/validation rows for this topology; increase data or adjust train-ratio for a smoke test')
        stage('preparing')
        checked('prepare',commands['prepare'])
        from datasets import load_from_disk
        # Audit the same values as integer lists, avoiding Tensor scalar loops.
        ds=load_from_disk(str(run/'prepared')).with_format(None)
        lengths=[];supervised=0
        for row in ds:
            ids=row['input_ids'];mask=row['loss_mask']
            assert 0<len(ids)<=a.sample_max_length and len(ids)==len(mask) and any(mask)
            lengths.append(len(ids));supervised+=sum(mask)
        if min(int(len(ds)*a.train_ratio),len(ds)-int(len(ds)*a.train_ratio))<a.replicas*a.shard_size:
            raise ValueError('Preparation left insufficient train/validation rows for all ranks')
        write(run/'prepared_audit.json',{'rows':len(ds),'input_tokens':sum(lengths),'supervised_tokens':int(supervised),'max_length':max(lengths)})
        prepared={str(p.relative_to(run/'prepared')):sha(p) for p in (run/'prepared').rglob('*') if p.is_file()}
        write(run/'prepared_fingerprints.json',prepared)
        assert_free('npu',all_cards)
        cache.mkdir(parents=True,exist_ok=False);write(cache/'OWNER.json',{'run':str(run),'identity':identity(os.getpid())})
        stage('starting_teachers')
        teachers=new_owner(run/'teachers')
        for i,(group,cmd) in enumerate(zip(resolved['teachers'],commands['teachers'])):
            write(run/f'teachers/server{i}_provenance.json',{'argv':cmd,'wrapper_consumed_provenance':True,'source_fingerprints':str(run/'protected_source.json')})
            p=teachers.start(f'server{i}',cmd,dict(env,ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,group))),cwd=source)
            teacher_processes.append(p);health(a.teacher_port+i,p)
        router_env=dict(env,TEACHER_ROUTER_PORT=str(a.router_port),TEACHER_BACKEND_PORTS=','.join(str(a.teacher_port+i) for i in range(len(resolved['teachers']))),
                        TEACHER_ROUTING_LOG=str(run/'router_routing.jsonl'))
        router=teachers.start('router',commands['router'],router_env,cwd=ROOT);health(a.router_port,router,60)
        stage('training',replicas=a.replicas,shard_size=a.shard_size,world_size=a.replicas*a.shard_size)
        checked('trainer',commands['trainer'],train_env)
        stage('verifying_training')
        from npu.verify_run import verify_training
        result=verify_training(run,a,resolved)
        assert protected_files(source)==protected
        assert all(sha(run/'prepared'/name)==value for name,value in prepared.items())
        for name,item in target.items():
            path=Path(name);assert path.stat().st_size==item['bytes'] and path.stat().st_mtime_ns==item['mtime_ns']
            if item['sha256']:assert sha(path)==item['sha256']
        result.update(source_and_prepared_fingerprints_unchanged=True,target_manifest_unchanged=True)
        write(run/'TRAINING_VERIFICATION.json',result);completed=True
    except BaseException:
        write(run/'failure.json',{'traceback':traceback.format_exc(),'utc_epoch':time.time()})
        raise
    finally:
        cleanup_error=None
        for owner in reversed(owners):
            try:owner.close()
            except BaseException as exc:cleanup_error=repr(exc)
        for owner in owners:
            cleanup=owner.run/'cleanup.json'
            if not cleanup.exists() or any(v['remaining'] for v in json.loads(cleanup.read_text())):
                cleanup_error='Missing or incomplete owned cleanup receipt: '+str(cleanup)
        write(run/'cleanup_summary.json',{'passed':cleanup_error is None,'error':cleanup_error})
        if cache.exists() and cleanup_error is None:
            marker=cache/'OWNER.json'
            if marker.exists() and json.loads(marker.read_text())['run']==str(run):
                assert cache.resolve()==(a.hidden_cache.resolve() if a.hidden_cache else run/'hidden_states')
                shutil.rmtree(cache)
                write(run/'hidden_cache_cleanup.json',{'removed':str(cache),'all_owned_consumers_stopped':True})
            else:
                cleanup_error='Hidden cache owner marker does not match this run'
        if completed and cleanup_error is None:
            result=json.loads((run/'TRAINING_VERIFICATION.json').read_text())
            result.update(passed=True,owned_processes_released=True,hidden_cache_removed=not cache.exists(),utc_epoch=time.time())
            write(run/'COMPLETION_VERIFICATION.json',result);stage('completed')
            print(json.dumps(result,indent=2))
        else:
            stage('failed')
            if completed:raise RuntimeError('Training finished but owned cleanup was not verified: '+str(cleanup_error))


if __name__=='__main__':main()
