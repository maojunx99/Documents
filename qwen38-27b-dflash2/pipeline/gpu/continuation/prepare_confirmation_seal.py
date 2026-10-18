# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Seal prompt-only confirmation after exact/near checks against history/pools.
No model request, score or generated response is produced by this script.
"""
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import traceback
from prepare_inventory import norm,sha,write,identity

WORD=re.compile(r'[\u3400-\u9fff]|[^\W_]+|_|[^\w\s]',re.UNICODE)


def shingles(q):
    t=WORD.findall(norm(q));return {'\x1f'.join(t[i:i+5]) for i in range(max(1,len(t)-4))}


def signature(parts):
    from datasketch import MinHash
    m=MinHash(num_perm=128,seed=2026100904)
    m.update_batch([x.encode() for x in parts]);return m


def init_worker(candidates,deps):
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,deps)
    from datasketch import MinHashLSH
    global LSH,SETS,EXACT
    LSH=MinHashLSH(num_perm=128,params=(32,4));SETS={};EXACT=collections.defaultdict(list)
    for r in candidates:
        k=r['candidate_id'];s=shingles(r['prompt']);SETS[k]=s;LSH.insert(k,signature(s));EXACT[r['prompt_sha256']].append(k)


def match_batch(rows):
    matches=[]
    for h,q in rows:
        if h in EXACT:
            matches.extend((k,h,1.0,'exact') for k in EXACT[h]);continue
        s=shingles(q);m=signature(s)
        for k in LSH.query(m):
            ref=SETS[k];sim=len(s&ref)/len(s|ref)
            if sim>=.85:matches.append((k,h,sim,'5gram_jaccard'))
    return len(rows),matches


def main(root,storage):
    began=time.time();deadline=began+7200
    write(root/'confirmation_identity.json',identity())
    sys.path.insert(0,str(root/'deps'))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/workspace/dflash2_work/models/Qwen3.8-27B',trust_remote_code=True)
    path=storage/'confirmation_sources/prompt_candidates.jsonl.gz'
    if not path.exists():raise RuntimeError('Verified prompt projection must be uploaded first')
    receipt=json.loads((storage/'confirmation_sources/prompt_projection_receipt.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    groups=collections.defaultdict(list)
    with gzip.open(path,'rt') as f:
        for line in f:
            r=json.loads(line);q=norm(r['prompt'])
            if len(q)<20:continue
            if r['dataset']=='numina':
                group='numina_basic' if r['source'] in {'cn_k12','orca_math','gsm8k'} else 'numina_advanced'
            else:group=r['dataset']
            r.update(group=group,prompt_sha256=sha(q),candidate_id=r['dataset']+'/'+r['id'])
            r['order_sha256']=sha('2026100904/'+r['candidate_id']+'/'+r['prompt_sha256'])
            groups[group].append(r)
    candidates=[]
    for group,rows in groups.items():
        for r in sorted(rows,key=lambda x:x['order_sha256'])[:1024 if group.startswith('numina') else len(rows)]:
            x=tok.apply_chat_template([dict(role='user',content=r['prompt'])],add_generation_prompt=True,tokenize=True,enable_thinking=True,reasoning_effort='xhigh')
            if hasattr(x,'get'):x=x['input_ids']
            if len(x)<=4096:r['prompt_tokens']=len(x);candidates.append(r)
    locked=storage/'confirmation_candidate_lock.json'
    if locked.exists():raise RuntimeError('Refusing to reselection candidate lock')
    write(locked,dict(seed=2026100904,candidates=candidates,threshold=.85,shingles='Unicode lexical/punctuation 5grams',minhash=dict(seed=2026100904,num_perm=128,bands=32,rows=4),scope='Candidate lock before history exclusion and before any model scoring'))
    counts=collections.Counter();dirty=set();seen=set();last_row=0
    def status(stage):
        if time.time()>deadline:raise TimeoutError('Confirmation preparation 2h cap')
        write(root/'confirmation_status.json',dict(stage=stage,time=time.time(),deadline=deadline,counts=dict(counts),dirty_candidates=len(dirty),candidates=len(candidates),optimizer_steps=0,model_requests=0))
    out=(root/'confirmation_contamination_hits.jsonl').open('x')
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=init_worker,initargs=(candidates,str(root/'deps'))) as pool:
        def consume(rows):
            chunks=[rows[i:i+256] for i in range(0,len(rows),256)]
            for n,matches in pool.map(match_batch,chunks):
                counts['unique_reference_prompts_checked']+=n
                for k,h,similarity,kind in matches:
                    dirty.add(k);out.write(json.dumps(dict(candidate=k,reference_sha256=h,jaccard=similarity,kind=kind))+'\n')
            out.flush()
        # Follow the write-once historical index by monotonically increasing rowid.
        history=storage/'history_prompts.sqlite'
        while True:
            if (root/'history_failure.json').exists():raise RuntimeError('Historical index failed; no confirmation seal')
            if not history.exists():status('waiting_history_index');time.sleep(5);continue
            db=sqlite3.connect('file:'+str(history)+'?mode=ro',uri=True)
            rows=db.execute('SELECT rowid,sha,prompt FROM prompts WHERE rowid>? ORDER BY rowid LIMIT 8192',(last_row,)).fetchall();db.close()
            if rows:
                last_row=rows[-1][0];fresh=[(h,q) for _,h,q in rows if h not in seen];seen.update(h for h,q in fresh);consume(fresh);status('checking_history_near_duplicates')
            elif (root/'history_index_report.json').exists():break
            else:status('waiting_history_append');time.sleep(5)
        for path in [storage/'pool_catalogue.sqlite']:
            db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);cur=db.execute('SELECT prompt_sha,prompt FROM prompts')
            while True:
                rows=cur.fetchmany(8192)
                if not rows:break
                fresh=[(h,q) for h,q in rows if h not in seen];seen.update(h for h,q in fresh);consume(fresh);status('checking_full_new_prompt_pool')
            db.close()
        local=storage/'local_eval_exclusions.jsonl.gz'
        if not local.exists():raise RuntimeError('Local historical exclusions missing; refusing seal')
        with gzip.open(local,'rt') as f:
            records=[json.loads(l) for l in f]
        rows=[(r['sha256'],r['prompt']) for r in records if r['sha256'] not in seen];consume(rows)
    out.close()
    from datasketch import MinHashLSH
    picked=[];used=set();lsh=MinHashLSH(num_perm=128,params=(32,4));sets={}
    quotas={'bigcodebench':160,'numina_basic':80,'numina_advanced':80,'ifeval':80}
    for group,quota in quotas.items():
        count=0
        for r in sorted([r for r in candidates if r['group']==group],key=lambda r:r['order_sha256']):
            if r['candidate_id'] in dirty or r['prompt_sha256'] in used:continue
            s=shingles(r['prompt']);sig=signature(s)
            if any(len(s&sets[k])/len(s|sets[k])>=.85 for k in lsh.query(sig)):continue
            picked.append(r);used.add(r['prompt_sha256']);sets[r['candidate_id']]=s;lsh.insert(r['candidate_id'],sig);count+=1
            if count==quota:break
        if count!=quota:raise RuntimeError(f'Not enough uncontaminated {group}: {count}/{quota}; version plan before any training')
    sealed=storage/'sealed_confirmation';sealed.mkdir(exist_ok=False)
    with (sealed/'cases.jsonl').open('x') as f:
        for r in picked:f.write(json.dumps(r,ensure_ascii=True)+'\n')
    seal=dict(count=len(picked),quotas=quotas,seed=2026100904,cases_sha256=hashlib.sha256((sealed/'cases.jsonl').read_bytes()).hexdigest(),candidate_lock_sha256=hashlib.sha256(locked.read_bytes()).hexdigest(),references_checked=counts['unique_reference_prompts_checked'],dirty_candidates=len(dirty),model_requests=0,unseal_gate='Development-selected candidate + independent training-seed replication only',limitations=['LSH candidate retrieval plus exact Jaccard>=0.85, not exhaustive semantic dedup','Does not establish absence from target pretraining','Projection first pinned Numina train shard only'])
    write(sealed/'SEAL.json',seal);write(root/'confirmation_seal_receipt.json',seal);status('confirmation_sealed_no_model_requests')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);args=a.parse_args()
    try:main(args.run,args.storage)
    except BaseException as e:
        write(args.run/'confirmation_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
