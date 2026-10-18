# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Bounded CPU candidate screening; no generation or training launch.

Each source is pinned. Outside solutions stay in the immutable raw snapshots.
The output remains quarantined until full prompt review and residual-budget lock.
"""
import argparse,collections,concurrent.futures,gzip,hashlib,json,sqlite3,sys,time,traceback
from pathlib import Path
import pyarrow.parquet as pq
from alternative_source_filters import reason
from prepare_inventory import identity,norm,sha,write
from prepare_confirmation_seal import init_worker,match_batch,shingles,signature


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(root,storage):
    start=time.time();deadline=start+7200
    out=storage/'alternative_candidates_02';out.mkdir(exist_ok=False)
    write(root/'alternative_candidates_v2_identity.json',identity())
    plan=json.loads((root/'DATA_SOURCE_ADDENDUM_v3.json').read_text())
    review=json.loads((root/'alternative_sources/manual_source_review.json').read_text())
    assert digest(root/'alternative_source_filters.py')==plan['filter_sha256']
    stats=collections.Counter();candidates=[];seen=set();dirty=set();reference_seen=set()
    def status(stage):
        if time.time()>deadline:raise TimeoutError('Alternative candidates CPU two-hour cap')
        write(root/'alternative_candidates_v2_status.json',dict(stage=stage,time=time.time(),deadline=deadline,counts=dict(stats),candidates=len(candidates),dirty_candidates=len(dirty),model_requests=0,optimizer_steps=0))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/workspace/dflash2_work/models/Qwen3.8-27B',trust_remote_code=True)
    sys.path.insert(0,str(root/'deps'))
    for item in plan['sources']:
        source=storage/'alternative_sources'/item['local_name']
        assert source.stat().st_size==item['bytes'] and digest(source)==item['sha256']
        isopen=item['local_name'].startswith('opencode');tag='opencode' if isopen else 'selfcodealign'
        rejects=review[tag+'_reject']
        cols=['id','input','domain','generation_algorithm','llm_judgement','average_test_score','tests_execution_status'] if isopen else ['id','instruction','concepts']
        group=[]
        for batch in pq.ParquetFile(source).iter_batches(batch_size=2048,columns=cols):
            for row in batch.to_pylist():
                stats[tag+'_source_rows']+=1;key=str(row['id']);q=row['input' if isopen else 'instruction']
                why='manual_reject' if key in rejects else reason(q,row,isopen)
                if why:stats[tag+'_reject_'+why]+=1;continue
                h=sha(norm(q))
                if h in seen:stats[tag+'_exact_duplicate']+=1;continue
                seen.add(h)
                group.append(dict(candidate_id=tag+'/'+key,id=key,source=item['repo'],revision=item['revision'],prompt=q,prompt_sha256=h,order_sha256=sha('2026100901/alternative/'+tag+'/'+key+'/'+h),metadata={k:v for k,v in row.items() if k not in ('input','instruction')},eligible_for_generation=False))
        group.sort(key=lambda x:x['order_sha256']);n=0
        for row in group:
            ids=tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')
            if hasattr(ids,'get'):ids=ids['input_ids']
            if len(ids)>1024:stats[tag+'_native_prompt_gt1024']+=1;continue
            row['prompt_tokens']=len(ids);candidates.append(row);n+=1
            if n==plan['source_candidate_cap_each']:break
        stats[tag+'_candidate_lock_rows']=n;status('structural_screen')
    with (out/'candidate_lock.jsonl').open('x') as f:
        for row in candidates:f.write(json.dumps(row,ensure_ascii=True)+'\n')
    status('checking_references')
    with (out/'contamination_hits.jsonl').open('x') as hits,concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=init_worker,initargs=(candidates,str(root/'deps'))) as pool:
        def consume(rows,kind):
            fresh=[]
            for h,q in rows:
                if h not in reference_seen:fresh.append((h,q));reference_seen.add(h)
            for count,matches in pool.map(match_batch,[fresh[i:i+256] for i in range(0,len(fresh),256)]):
                stats['unique_reference_prompts_checked']+=count
                for key,h,sim,method in matches:
                    dirty.add(key);hits.write(json.dumps(dict(candidate=key,reference_sha256=h,jaccard=sim,method=method,reference_kind=kind))+'\n')
            hits.flush();status('checking_'+kind)
        # Never use confirmation answers or model performance to choose training data.
        refs=json.loads((storage/'confirmation_candidate_lock.json').read_text())['candidates']
        consume([(r['prompt_sha256'],r['prompt']) for r in refs],'confirmation_candidate_pool')
        for file,sql,kind in [
            ('history_prompts.sqlite','SELECT sha,prompt FROM prompts','full_historical_training_and_evaluation'),
            ('pool_catalogue.sqlite','SELECT prompt_sha,prompt FROM prompts','existing_pool')]:
            db=sqlite3.connect('file:'+str(storage/file)+'?mode=ro',uri=True);cursor=db.execute(sql)
            while True:
                rows=cursor.fetchmany(8192)
                if not rows:break
                consume(rows,kind)
            db.close()
        with gzip.open(storage/'local_eval_exclusions.jsonl.gz','rt') as f:refs=[json.loads(l) for l in f]
        consume([(r['sha256'],r['prompt']) for r in refs],'local_evaluation')
        refs=[json.loads(l) for l in (storage/'replay_index.jsonl').read_text().splitlines()]
        consume([(r['prompt_sha256'],r['prompt']) for r in refs],'replay')
        refs=[json.loads(l) for l in (storage/'fallback_generation_prompts_locked.jsonl').read_text().splitlines()]
        consume([(r['prompt_sha256'],r['prompt']) for r in refs],'active_magicoder_prompt_lock')
    from datasketch import MinHashLSH
    lsh=MinHashLSH(num_perm=128,params=(32,4));sets={};picked=[]
    for row in sorted(candidates,key=lambda x:x['order_sha256']):
        if row['candidate_id'] in dirty:continue
        s=shingles(row['prompt']);m=signature(s)
        if any(len(s&sets[k])/len(s|sets[k])>=.85 for k in lsh.query(m)):
            stats['internal_near_duplicate']+=1;continue
        key=row['candidate_id'];sets[key]=s;lsh.insert(key,m);picked.append(row)
    path=out/'decontaminated_candidates.jsonl'
    with path.open('x') as f:
        for row in picked:f.write(json.dumps(row,ensure_ascii=True)+'\n')
    reviews={source:[r for r in picked if r['source']==source][:40] for source in {r['source'] for r in picked}}
    write(root/'alternative_candidates_v2_review40.json',dict(reviewed=False,sources=reviews))
    result=dict(stage='candidates_decontaminated_prompt_review_pending',counts=dict(stats),candidate_lock_rows=len(candidates),dirty_candidates=len(dirty),retained=len(picked),retained_per_source=dict(collections.Counter(r['source'] for r in picked)),candidate_file=str(path),sha256=digest(path),method='NFKC whitespace exact; lexical5gram MinHash128/32x4 retrieval with exact Jaccard>=.85 verification',duration_s=time.time()-start,model_requests=0,optimizer_steps=0,not_training_ready=True,limitations=['LSH is not exhaustive semantic decontamination','Pinned OpenCode source first shard only','Generated target responses and quality/length/token quota gates pending','No additional generation budget granted'])
    write(root/'alternative_candidates_v2_report.json',result);write(root/'alternative_candidates_v2_status.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--storage',type=Path,required=True);args=ap.parse_args()
    try:main(args.run,args.storage)
    except BaseException as e:
        write(args.run/'alternative_candidates_v2_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
