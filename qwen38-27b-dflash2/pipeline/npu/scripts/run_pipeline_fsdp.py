# Publication changes: configurable model/data/Python; bundled preflights; fresh-preparation and target manifests.
# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Finite, fresh-only 10k NPU scratch workflow. Run inside the audited container."""
import argparse, hashlib, json, os, pathlib, re, shutil, signal, subprocess, sys, time, traceback, urllib.request

BASE=pathlib.Path(__file__).resolve().parents[1]
PY=pathlib.Path(os.environ.get('DFLASH_PYTHON', sys.executable))
MODEL=pathlib.Path(os.environ.get('DFLASH_TARGET_MODEL', '/models/Qwen3.8-27B'))
DATA=pathlib.Path(os.environ.get('DFLASH_REGENERATED_DATA', '/data/regenerated/shards'))
SOURCE=BASE/'source'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def ident(pid):
    try:
        stat=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'starttime':stat[19],'pgid':int(stat[2]),'state':stat[0],
                'cmdline':pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=pathlib.Path)
    ap.add_argument('--teacher-devices',default='2,3');ap.add_argument('--train-devices',default='8,9,10,11')
    ap.add_argument('--teacher-port',default=8352,type=int);ap.add_argument('--max-seconds',default=21600,type=int)
    ap.add_argument('--max-anchors',default=512,type=int)
    ap.add_argument('--max-steps',type=int)
    ap.add_argument('--selected-input',type=pathlib.Path)
    ap.add_argument('--prepared-input',type=pathlib.Path)
    ap.add_argument('--preflight-input',type=pathlib.Path)
    a=ap.parse_args();run=a.run.resolve()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert run.parent == BASE/'runs', 'Use a new run directory under this bundle npu/runs'
    assert MODEL.is_dir(), MODEL
    run.mkdir(parents=True,exist_ok=False)
    target_files={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 1024*1024 else None}
                  for p in MODEL.iterdir() if p.is_file()}
    save(run/'target_model_manifest.json',target_files)
    env=os.environ.copy();cann_python=pathlib.Path(env.get('ASCEND_HOME_PATH','/usr/local/Ascend/cann-9.1.0'))/'python/site-packages'
    assert (cann_python/'acl').exists(),str(cann_python)
    env.update(PYTHONPATH=f'{SOURCE}/hs_connectors/src:{SOURCE}/src:{BASE}/scripts:{cann_python}:'+env.get('PYTHONPATH',''),
        TOKENIZERS_PARALLELISM='false',TORCHDYNAMO_DISABLE='1',TASK_QUEUE_ENABLE='1',
        HCCL_EXEC_TIMEOUT='7200',HCCL_NPU_SOCKET_PORT_RANGE='46000-49000',OMP_NUM_THREADS='4')
    started=time.time();owned=[];children=[]
    save(run/'launch_identity.json',ident(os.getpid()))
    shutil.copytree(BASE/'scripts',run/'implementation_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    save(run/'implementation_manifest.json',{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (BASE/'scripts').glob('*.py')})
    config={'target':str(MODEL),'source':str(SOURCE),'source_commit':'7a58fc56217632d8d179b665734fa2269e8d9ffa',
      'rows':10000,'initialization':'random draft; pretrained frozen target embedding/head; no public draft',
      'epochs':1,'packed_tokens_per_rank':4096,'prepared_sample_max':4096,'anchors_per_packed_rank':a.max_anchors,
      'draft_config':str(BASE/'draft_config.json'),'block_size':8,'selector_rank':256,'selector_top_k':16,
      'selector_alpha':.1,'unary_ce':.1,'unary_tv':.9,'gamma':4.,'noise_std':.05,'seed':42,
      'optimizer':'AdamW','lr':.0006,'weight_decay':0.,'scheduler':'cosine','warmup_ratio':.04,
      'split':.99,'teacher_devices':a.teacher_devices,'training_devices':a.train_devices,'native_attn':'sdpa',
      'fsdp_shard':True,'world_size':4,'max_steps':a.max_steps,'gradient_checkpointing':True,'checkpoint_use_reentrant':False,'zero_grad_before_forward':True,
      'task_queue_enable':1,
      'checkpoint_dtype_conversion_device':'cpu',
      'difference_from_GPU_recipe':f'Workflow validation uses one epoch, packed4096/{a.max_anchors}anchors instead of8192/1024; same draft architecture and objectives. 512anchors exhausted64GiB NPU memory in run05.',
      'no_claim':'This verifies pipeline execution, not converged model quality or exact GPU arithmetic parity.'}
    save(run/'TRAINING_CONFIG.json',config)
    def status(stage,**more):save(run/'status.json',{'stage':stage,'utc_epoch':time.time(),**more})
    def launch(name,argv,extra=None):
        e=env.copy();e.update(extra or {});log=(run/(name+'.log')).open('w')
        p=subprocess.Popen([str(x) for x in argv],stdout=log,stderr=subprocess.STDOUT,env=e,cwd=SOURCE,start_new_session=True)
        identity=ident(p.pid);owned.append(identity);children.append(p)
        save(run/(name+'_identity.json'),identity);save(run/'owned_leaders.json',owned)
        save(run/(name+'_command.json'),{'argv':[str(x) for x in argv],'environment_overrides':{k:e[k] for k in ['PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES','TORCHDYNAMO_DISABLE'] if k in e}})
        return p
    def wait(p,limit):
        deadline=min(started+a.max_seconds,time.time()+limit)
        while p.poll() is None:
            if time.time()>deadline:raise TimeoutError('Stage deadline exceeded')
            time.sleep(2)
        if p.returncode:raise RuntimeError(f'Process{p.pid} exit{p.returncode}')
    try:
        inventory=subprocess.check_output(['npu-smi','info'],text=True)
        (run/'npu_before.txt').write_text(inventory)
        selected={int(x) for x in (a.teacher_devices+','+a.train_devices).split(',')}
        assert len(selected)==len(a.teacher_devices.split(','))+len(a.train_devices.split(','))
        for line in inventory.splitlines():
            m=re.match(r'\|\s*(\d+)\s+\d+\s+(\d+)\s*\|',line)
            if m and int(m[1]) in selected:raise RuntimeError('Selected NPU is occupied: '+line)
            m=re.match(r'\|\s*(\d+)\s+910\S*\s*\|\s*(\S+)',line)
            if m and int(m[1]) in selected and m[2]!='OK':raise RuntimeError('Selected NPU health: '+line)
        # Selection and preparation touch only this fresh run directory.
        assert a.max_anchors==512,'This workflow is locked to512anchors'
        assert json.loads((BASE/'probes/toy_01/status.json').read_text())['stage']=='passed'
        status('selecting_10000')
        if a.selected_input is None:
            wait(launch('select',[PY,BASE/'scripts/select_10k.py','--source',DATA,'--output',run/'selected']),900)
        else:
            selected=a.selected_input.resolve();manifest=json.loads((selected/'DATA_MANIFEST.json').read_text())
            assert manifest['rows']==10000
            assert hashlib.sha256((selected/'selected_10000.jsonl').read_bytes()).hexdigest()==manifest['output_sha256']
            (run/'selected').symlink_to(selected,target_is_directory=True)
            save(run/'selection_reuse.json',{'from':str(selected),'sha256':manifest['output_sha256'],'rows':10000})
        status('preparing_10000')
        if a.prepared_input is None:
            wait(launch('prepare',[PY,SOURCE/'scripts/prepare_data.py','--model',MODEL,'--data',run/'selected/selected_10000.jsonl',
                '--seq-length','4096','--output',run/'prepared','--seed','42','--num-preprocessing-workers','8']),1800)
        else:
            prepared=a.prepared_input.resolve()
            (run/'prepared').symlink_to(prepared,target_is_directory=True)
            save(run/'preparation_reuse.json',{'from':str(prepared),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared.iterdir() if p.is_file()}})
        save(run/'prepared_fingerprints.json', {'files': {str(p.relative_to(run/'prepared')): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (run/'prepared').rglob('*') if p.is_file()}})
        from datasets import load_from_disk
        ds=load_from_disk(str(run/'prepared'));assert len(ds)==10000
        lengths=[];supervised=[]
        for row in ds:
            ids=row['input_ids'];mask=row['loss_mask']
            nonempty=bool(mask.any()) if hasattr(mask,'any') else any(mask)
            count=int(mask.sum()) if hasattr(mask,'sum') else sum(mask)
            assert len(ids)==len(mask) and 0<len(ids)<=4096 and nonempty
            lengths.append(len(ids));supervised.append(count)
        save(run/'audit/prepared.json',{'rows':len(ds),'input_tokens':sum(lengths),'supervised_tokens':sum(supervised),'max_length':max(lengths)})
        status('npu_loss_attention_preflight')
        reused_hccl=False
        if a.preflight_input is None:
            wait(launch('kernel_preflight',[PY,BASE/'scripts/preflight_npu.py'],
                {'ASCEND_RT_VISIBLE_DEVICES':a.train_devices.split(',')[0],'AUDIT_OUT':str(run/'audit/npu_kernel_preflight.json')}),900)
        else:
            previous=a.preflight_input.resolve();old=json.loads((previous/'implementation_manifest.json').read_text())
            name='preflight_npu.py'
            assert hashlib.sha256((BASE/'scripts'/name).read_bytes()).hexdigest()==old['scripts/'+name]
            assert json.loads((previous/'audit/npu_kernel_preflight.json').read_text())['passed']
            shutil.copy2(previous/'audit/npu_kernel_preflight.json',run/'audit/npu_kernel_preflight.json')
            old_cfg=json.loads((previous/'TRAINING_CONFIG.json').read_text())
            expected_world=len(a.train_devices.split(','))
            candidates=[previous/f'audit/hccl_preflight.rank{rank}.json' for rank in range(expected_world)]
            if old_cfg['training_devices']==a.train_devices and all(p.exists() for p in candidates) and hashlib.sha256((BASE/'scripts/preflight_hccl.py').read_bytes()).hexdigest()==old['scripts/preflight_hccl.py']:
                if all(json.loads(p.read_text()).get('world_size')==expected_world for p in candidates):
                    for p in candidates:shutil.copy2(p,run/'audit'/p.name)
                    reused_hccl=True
            save(run/'preflight_reuse.json',{'from':str(previous),'kernel_reused':True,'full_topology_hccl_reused':reused_hccl,'reason':'Kernel/source/packages unchanged. HCCL is rerun unless both implementation and entire rank/device topology match.'})
        if not reused_hccl:
            status('hccl_full_topology_preflight')
            wait(launch('hccl_preflight',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(a.train_devices.split(','))),BASE/'scripts/preflight_hccl.py'],
                {'ASCEND_RT_VISIBLE_DEVICES':a.train_devices,'AUDIT_OUT':str(run/'audit/hccl_preflight')}),300)
        status('starting_owned_teacher')
        checkpointing=json.loads((pathlib.Path(str(BASE/'preflight/CHECKPOINTING_PREFLIGHT.json'))).read_text())
        assert checkpointing['cpu']['passed'] and checkpointing['npu']['passed']
        shutil.copy2(pathlib.Path(str(BASE/'preflight/CHECKPOINTING_PREFLIGHT.json')),run/'audit/checkpointing_preflight.json')
        cpu_checkpoint=json.loads((pathlib.Path(str(BASE/'preflight/CPU_CHECKPOINT_PREFLIGHT.json'))).read_text())
        assert cpu_checkpoint['passed']
        shutil.copy2(pathlib.Path(str(BASE/'preflight/CPU_CHECKPOINT_PREFLIGHT.json')),run/'audit/cpu_checkpoint_preflight.json')
        (run/'teacher_provenance').mkdir();(run/'hidden_states').mkdir()
        teacher_args=[PY,SOURCE/'scripts/launch_vllm.py',MODEL,'--target-layer-ids','6','20','34','48','62',
          '--hidden-states-path',run/'hidden_states','--','--tensor-parallel-size','2','--port',str(a.teacher_port),
          '--dtype','bfloat16','--enforce-eager','--no-enable-chunked-prefill','--gpu-memory-utilization','.80',
          '--max-num-seqs','4','--max-num-batched-tokens','8192','--max-model-len','8192','--renderer-num-workers','1']
        # Pinned native launcher has no provenance CLI. This wrapper consumes and
        # implements the required provenance directory before invoking native.
        save(run/'teacher_provenance/vllm_command.json',{'argv':[str(x) for x in teacher_args],'provenance_dir':str(run/'teacher_provenance'),
          'wrapper_consumed_provenance':True,'source_manifest':str(BASE/'SOURCE_MANIFEST.json')})
        teacher=launch('teacher',teacher_args,{'ASCEND_RT_VISIBLE_DEVICES':a.teacher_devices})
        deadline=time.time()+900
        while True:
            if teacher.poll() is not None:raise RuntimeError('Teacher failed before readiness')
            try:
                if urllib.request.urlopen(f'http://127.0.0.1:{a.teacher_port}/health',timeout=3).status==200:break
            except Exception:pass
            if time.time()>deadline:raise TimeoutError('Teacher readiness')
            time.sleep(3)
        save(run/'teacher_provenance/health.json',{'health':200,'utc_epoch':time.time()})
        status('training_epoch0')
        args=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(a.train_devices.split(','))),BASE/'scripts/run_train_fsdp.py',
          '--verifier-name-or-path',MODEL,'--speculator-type','dflash2','--draft-config',BASE/'draft_config.json','--draft-attn-impl','sdpa',
          '--data-path',run/'prepared','--hidden-states-backend','file','--hidden-states-path',run/'hidden_states',
          '--vllm-endpoint',f'http://127.0.0.1:{a.teacher_port}/v1','--save-path',run/'checkpoints','--log-dir',run/'logs',
          '--run-name','qwen38_27b_npu_scratch10k','--block-size','8','--max-anchors',str(a.max_anchors),'--target-layer-ids','6','20','34','48','62',
          '--mask-token-id','248070','--conv-kernel-size','2','--conv-group-size','16','--selector-rank','256','--selector-top-k','16',
          '--selector-loss-alpha','.1','--loss-fn','{"ce":0.1,"tv":0.9}','--loss-implementation','fused',
          '--dflash-decay-gamma','4','--per-position-loss-weight','fixed-exp-decay','--optimizer','adamw','--lr','.0006','--weight-decay','0',
          '--scheduler-type','cosine','--scheduler-warmup-ratio','.04','--epochs','1','--total-seq-len','4096','--train-data-ratio','.99',
          '--seed','42','--num-workers','4','--prefetch-factor','2','--noise-std','.05','--hidden-states-dtype','bfloat16',
          '--request-timeout','900','--max-retries','3','--on-missing','generate','--on-generate','delete','--checkpoint-freq','1','--fsdp-shard',
          '--log-freq','20','--no-resume-from-checkpoint']
        if a.max_steps is not None:args.extend(['--max-steps',str(a.max_steps)])
        trainer=launch('trainer',args,{'ASCEND_RT_VISIBLE_DEVICES':a.train_devices,'TRAIN_AUDIT':str(run/'audit'),'NATIVE_SOURCE':str(SOURCE)})
        wait(trainer,a.max_seconds)
        status('training_and_native_validation_completed')
        ckpt=run/'checkpoints/0';assert (ckpt/'model.safetensors').exists()
        initial=[json.loads(p.read_text()) for p in sorted((run/'audit').glob('initial_rank*.json'))]
        assert len(initial)==len(a.train_devices.split(','))
        assert all(x['trainable']==initial[0]['trainable'] for x in initial)
        for rank,item in enumerate(initial):
            updates=[json.loads(l) for l in (run/f'audit/updates_rank{rank}.jsonl').read_text().splitlines()]
            assert len(updates)==min(item['train_steps'],a.max_steps or item['train_steps'])
            assert updates[-1]['sample_weight_changed_since_initialization']
            assert (run/f'audit/validation_epoch0_rank{rank}.json').exists()
        save(run/'training_completion.json',{'passed':True,'ranks':len(initial),'steps_per_rank':[min(x['train_steps'],a.max_steps or x['train_steps']) for x in initial],
          'checkpoint':str(ckpt),'model_sha256':hashlib.sha256((ckpt/'model.safetensors').read_bytes()).hexdigest(),'utc_epoch':time.time()})
        assert json.loads((run/'audit/checkpoint_reload_rank0.json').read_text())['passed']
        status('native_training_validation_and_reload_complete_cleanup_pending')
    except BaseException:
        save(run/'failure.json',{'traceback':traceback.format_exc(),'utc_epoch':time.time()});status('failed');raise
    finally:
        cleanup=[]
        for p,identity in reversed(list(zip(children,owned))):
            now=ident(p.pid)
            if now and now['state']!='Z' and now['starttime']==identity['starttime'] and now['pgid']==identity['pgid']:
                os.killpg(identity['pgid'],signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    now=ident(p.pid)
                    if now and now['starttime']==identity['starttime']:os.killpg(identity['pgid'],signal.SIGKILL)
                    p.wait(timeout=15)
            cleanup.append({'identity':identity,'after':ident(p.pid)})
        save(run/'cleanup.json',cleanup)

if __name__=='__main__':main()
