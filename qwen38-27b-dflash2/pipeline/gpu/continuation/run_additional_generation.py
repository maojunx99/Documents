# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Bounded target-only regeneration. Four free TP1 GPUs, exact tokens, no tools.

Requires a reviewed prompt lock. Never starts training or replaces source data.
Direct vLLM serving is used (not the hidden-state launch_vllm train command);
full argv/environment/source/model fingerprints are persisted explicitly.
"""
import argparse,asyncio,fcntl,hashlib,json,os,signal,socket,subprocess,time,traceback,urllib.request
from pathlib import Path
from generation_budget import GenerationBudget
from prepare_inventory import write

W=Path('/workspace/dflash2_work')
TARGET=W/'models/Qwen3.8-27B';PY=str(W/'venv/bin/python')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def ident(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,starttime=s[19],state=s[0],ppid=int(s[1]),pgid=int(s[2]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None

def alive(rec):
    x=ident(rec['pid']);return bool(x and x['starttime']==rec['starttime'] and x['state']!='Z')

def owned_tree(rec):
    x=ident(rec['pid'])
    if not alive(rec):return []
    assert x['cmdline']==rec['cmdline'] and x['pgid']==rec['pgid']
    found={x['pid']:x}
    for _ in range(12):
        added=False
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) in found:continue
            x=ident(int(p.name))
            if x and x['ppid'] in found:found[x['pid']]=x;added=True
        if not added:break
    return list(found.values())

def cleanup(out,services):
    rows=[x for r in services for x in owned_tree(r)];write(out/'cleanup_before.json',rows)
    for x in rows:
        now=ident(x['pid'])
        if alive(x):
            assert now['cmdline']==x['cmdline'] and now['pgid']==x['pgid'];os.kill(x['pid'],signal.SIGTERM)
    until=time.monotonic()+45
    while any(alive(x) for x in rows) and time.monotonic()<until:time.sleep(1)
    remaining=[x for x in rows if alive(x)]
    write(out/'cleanup.json',dict(observed=rows,remaining=remaining))
    if remaining:raise RuntimeError('Owned generation servers did not exit; inspect exact identities')

def available_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={x.split(',')[0].strip() for x in apps.splitlines() if ',' in x}
    return [int(a) for a,u,m in (x.split(',') for x in raw.splitlines()) if float(m)<128 and u.strip() not in busy]

async def generate(out,storage,rows,ports,deadline,tok):
    import aiohttp
    from generation_budget import GenerationBudget
    limits=json.loads((out.parent/"RESIDUAL_BUDGET.json").read_text());budget=GenerationBudget(rows=limits["max_prompts"],tokens=limits["max_completion_tokens"]);gate=asyncio.Lock();next_index=0;fatal=[]
    cases=storage/'fallback_generation';cases.mkdir(exist_ok=False)
    (cases/'cases').mkdir();sink=(cases/'responses.jsonl').open('x')
    def status(stage):write(out/'status.json',dict(stage=stage,time=time.time(),deadline=deadline,budget=budget.snapshot(),rows=len(rows),fatal=fatal,optimizer_steps=0))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def one(index,port):
            row=rows[index];key=row['id'];d=cases/'cases'/key;d.mkdir()
            seed=int(hashlib.sha256(('2026100901/generation/'+key).encode()).hexdigest()[:8],16)&0x7fffffff
            payload=dict(model=str(TARGET),messages=[dict(role='user',content=row['prompt'])],temperature=1.0,top_p=.95,top_k=20,presence_penalty=0.,reasoning_effort='xhigh',chat_template_kwargs=dict(enable_thinking=True),max_tokens=8192,seed=seed,return_token_ids=True)
            write(d/'request.json',payload)
            before=time.time()
            remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError('Generation wall budget reached')
            async with session.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=payload,timeout=aiohttp.ClientTimeout(total=min(900,remaining))) as res:
                raw=await res.read();(d/'response.raw.json').write_bytes(raw)
                if res.status!=200:raise RuntimeError(f'{key}: HTTP {res.status}')
            data=json.loads(raw);choice=data['choices'][0];prompt=data['prompt_token_ids'];completion=choice['token_ids']
            expected=tok.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')
            if hasattr(expected,'get'):expected=expected['input_ids']
            assert prompt==expected, 'Native prompt/template IDs mismatch'
            assert data['usage']['prompt_tokens']==len(prompt)
            assert data['usage']['completion_tokens']==len(completion), 'Usage/token IDs mismatch'
            assert 0<len(completion)<=8192 and choice['finish_reason'] in ('stop','length')
            assert not choice['message'].get('tool_calls'), 'No tool execution/generation requested'
            sample=dict(id=key+'_gen0',primary_id=key,input_ids=prompt+completion,loss_mask=[0]*len(prompt)+[1]*len(completion),text=tok.decode(prompt+completion,skip_special_tokens=False),metadata=dict(idx=index,source=row['source'],source_revision=row['revision'],prompt_sha256=row['prompt_sha256'],finish_reason=choice['finish_reason'],is_tool_call=False,usage=data['usage'],endpoint=f'http://127.0.0.1:{port}/v1/chat/completions',sampling_params={k:payload[k] for k in ['temperature','top_p','top_k','presence_penalty','reasoning_effort','max_tokens','seed']},elapsed_s=time.time()-before,raw_response_sha256=hashlib.sha256(raw).hexdigest()))
            async with gate:
                budget.finish(key,len(completion));sink.write(json.dumps(sample,ensure_ascii=True)+'\n');sink.flush();status('generating')
            write(d/'verification.json',dict(prompt_tokens_match=True,usage_tokens_match=True,manual_system_messages=0,seed=seed,finish_reason=choice['finish_reason'],completion_tokens=len(completion)))
        # First real request is part of the fixed pool and budget, never a repeat.
        assert rows and budget.reserve(rows[0]['id']);next_index=1
        await one(0,ports[0]);write(out/'first_request_verification.json',dict(passed=True,id=rows[0]['id'],budget=budget.snapshot(),time=time.time()))
        async def worker(port):
            nonlocal next_index
            while True:
                async with gate:
                    if fatal or time.time()>=deadline or next_index>=len(rows):return
                    index=next_index;key=rows[index]['id']
                    if budget.reserve(key):next_index+=1
                    elif not budget.pending:return
                    else:index=None
                if index is None:await asyncio.sleep(.2);continue
                try:await one(index,port)
                except BaseException as e:
                    async with gate:
                        fatal.append(dict(index=index,id=key,error=repr(e)));status('request_failed_waiting_inflight')
                    return
        await asyncio.gather(*(worker(port) for port in ports for _ in range(16)))
    sink.close()
    if fatal:raise RuntimeError(str(fatal))
    assert not budget.pending
    result=dict(stage='generated_pending_quality_and_mixture_selection',budget=budget.snapshot(),available_prompts=len(rows),time=time.time(),deadline=deadline,responses_sha256=digest(cases/'responses.jsonl'),stop_reason='all_prompts' if next_index==len(rows) else 'wall_limit' if time.time()>=deadline else 'completion_token_budget',optimizer_steps=0)
    write(out/'result.json',result);write(out/'status.json',result)

def main(root,storage,wait_seconds):
    out=root/'generation';out.mkdir(exist_ok=False)
    lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(out/'launch_identity.json',ident(os.getpid()))
    review=json.loads((root/'fallback_prompt_audit_v2.json').read_text());assert review['passed']
    selection=storage/'fallback_generation_prompts_locked.jsonl';assert digest(selection)==review['generation_prompts_sha256']
    rows=[json.loads(l) for l in selection.read_text().splitlines()];assert 0<len(rows)<=10000
    assert (root/'confirmation_seal_receipt.json').exists()
    limits=json.loads((root/"RESIDUAL_BUDGET.json").read_text());assert limits["absolute_deadline"]>time.time()+1200
    wait_end=min(time.time()+wait_seconds,limits["absolute_deadline"]-1200);stable=None
    while True:
        free=available_gpus()
        if len(free)>=4:
            chosen=tuple(free[:4])
            if stable and stable[0]==chosen and time.time()-stable[1]>=15:break
            if not stable or stable[0]!=chosen:stable=(chosen,time.time())
        else:stable=None
        write(out/'status.json',dict(stage='waiting_four_exclusively_free_gpus',free_gpus=free,time=time.time(),wait_deadline=wait_end,model_requests=0,optimizer_steps=0))
        if time.time()>=wait_end:raise TimeoutError('No four exclusively free GPUs within bounded resource wait; no servers started')
        time.sleep(5)
    ports=[8381,8382,8383,8384]
    for p in ports:
        with socket.socket() as s:s.settimeout(1);assert s.connect_ex(('127.0.0.1',p))!=0
    # Hash target once before use; later compare after owned processes exit.
    files=sorted(p for p in TARGET.iterdir() if p.is_file())
    fingerprints={str(p):digest(p) for p in files}
    write(out/'target_fingerprints_before.json',fingerprints)
    import importlib.metadata
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(TARGET,trust_remote_code=True)
    deadline=limits["absolute_deadline"];assert deadline>time.time()+1200;services=[];processes=[]
    try:
        assert all(g in available_gpus() for g in chosen), 'GPU allocation changed during provenance hashing'
        for gpu,port in zip(chosen,ports):
            d=out/f'gpu{gpu}';d.mkdir()
            cmd=[PY,'-m','vllm.entrypoints.cli.main','serve',str(TARGET),'--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','10240','--max-num-seqs','32','--gpu-memory-utilization','.80','--generation-config','vllm','--mm-processor-cache-gb','0','--port',str(port)]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            write(d/'provenance.json',dict(argv=cmd,env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']},versions={k:importlib.metadata.version(k) for k in ['torch','vllm','transformers']},runner_sha256=digest(__file__),target_fingerprints=fingerprints,manual_system_messages=0))
            with (d/'server.log').open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            x=ident(p.pid);services.append(x);processes.append(p);write(d/'identity.json',x);write(out/'server_identities.json',services)
        startup=min(time.time()+1200,deadline)
        while True:
            assert all(p.poll() is None for p in processes),'Generation server exited during startup'
            ready=[]
            for port in ports:
                try:ready.append(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).status==200)
                except Exception:ready.append(False)
            write(out/'status.json',dict(stage='starting_generation_servers',ready=ready,time=time.time(),deadline=deadline,optimizer_steps=0))
            if all(ready):break
            if time.time()>=startup:raise TimeoutError('Generation server startup deadline')
            time.sleep(5)
        asyncio.run(generate(out,storage,rows,ports,deadline,tok))
    finally:
        cleanup(out,services)
    after={p:digest(p) for p in fingerprints};assert after==fingerprints
    write(out/'completion_verification.json',dict(target_fingerprints_unchanged=True,owned_servers_exited=True,optimizer_steps=0,time=time.time(),response_result=json.loads((out/'result.json').read_text())))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);a.add_argument('--wait-seconds',type=int,default=7200);x=a.parse_args()
    try:main(x.run,x.storage,x.wait_seconds)
    except BaseException as e:
        out=x.run/'generation';out.mkdir(exist_ok=True);write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=ident(os.getpid())));raise
