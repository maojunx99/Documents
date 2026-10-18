# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Version-3 deterministic cluster split, mixture, Arrow and native schedule.

No main training is launched. A lock is written only after all assertions pass.
Failure leaves partial artifacts for diagnosis, never a usable selection lock.
"""
import argparse, collections, hashlib, json, math, sys, time, traceback
from pathlib import Path
import sqlite3
import numpy as np
from prepare_inventory import write,identity,sha,norm


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root,storage):
    began=time.time();write(root/'selection3_identity.json',identity())
    out=storage/'selection_v3_03';out.mkdir(exist_ok=False)
    plan=json.loads((root/'PLAN_DATA_SCALE_REVISION_v3.json').read_text())
    report=json.loads((root/'cluster_candidates_report.json').read_text())
    projected=storage/'cluster_candidates_v1/candidates.jsonl'
    assert digest(projected)==report['candidate_sha256']==plan['evidence']['raw_candidate_manifest_sha256']
    groups=collections.defaultdict(list)
    for line in projected.read_text().splitlines():
        r=json.loads(line);groups[r['cluster_sha256']].append(r)
    reps=[min(rr,key=lambda x:(x['bucket']!='replay',x['sort_sha'])) for rr in groups.values()]
    buckets=collections.defaultdict(list)
    for r in reps:buckets[r['bucket']].append(r)
    val=[];reserved=set()
    for bucket,count in plan['offline_validation_bucket_counts'].items():
        rows=sorted(buckets[bucket],key=lambda r:sha('2026100903/offline/'+r['cluster_sha256']))
        assert len(rows)>=count
        val.extend(rows[:count]);reserved.update(r['cluster_sha256'] for r in rows[:count])
    assert len(val)==1000 and len(reserved)==1000
    prior_split=json.loads((storage/'selection_v3/OFFLINE_SPLIT_LOCK.json').read_text())
    assert prior_split['clusters']==sorted(reserved), 'Recovery changed locked validation clusters'
    write(out/'OFFLINE_SPLIT_LOCK.json',dict(rows=1000,clusters=sorted(reserved),seed=2026100903,
        basis='All selected-candidate prompt/answer/primary-function clusters formed before split; no model scores',
        per_bucket=plan['offline_validation_bucket_counts']))
    train=[];used=set(reserved);totals=collections.Counter();source_totals=collections.Counter();flags=collections.Counter()
    target=plan['scope']['target_pool_supervised_tokens']
    def add(r,b):
        n=r['supervised']
        if r['cluster_sha256'] in used:return False
        if flags['contest']+n*r['contest']>target*.049:return False
        if flags['easy_math']+n*r['easy_math']>target*.049:return False
        if b=='general_instruction' and source_totals[r['source']]+n>target*.15:return False
        train.append(r);used.add(r['cluster_sha256']);totals[b]+=n;source_totals[r['source']]+=n
        for flag in ['contest','easy_math','complete','python']:flags[flag]+=n*bool(r[flag])
        return True
    # Replay first to reserve its source-stratified allowance in global caps.
    rb=collections.defaultdict(list)
    for r in buckets['replay']:rb[r['source']].append(r)
    for rows in rb.values():rows.sort(key=lambda r:r['sort_sha'])
    while totals['replay']<plan['bucket_targets']['replay']:
        added=False
        for source in sorted(rb):
            while rb[source]:
                if add(rb[source].pop(0),'replay'):added=True;break
            if totals['replay']>=plan['bucket_targets']['replay']:break
        if not added:raise RuntimeError('Insufficient distinct replay after split/caps')
    for b in ['short_function','code_reasoning_and_repair','general_instruction','math']:
        rows=sorted(buckets[b],key=lambda r:r['sort_sha'])
        goal=plan['bucket_targets'][b]
        if b=='math':
            for r in rows:
                if r['tokens']>2048 and not r['easy_math']:add(r,b)
                if totals[b]>=goal*.40:break
            if totals[b]<goal*.40:raise RuntimeError('Insufficient complete long non-easy math')
        for r in rows:
            if totals[b]>=goal:break
            add(r,b)
        if totals[b]<goal:raise RuntimeError(f'Insufficient {b}: {totals[b]} < {goal}')
    train.sort(key=lambda r:r['sort_sha']);val.sort(key=lambda r:sha('2026100903/'+r['key']))
    assert len({r['prompt_sha256'] for r in train+val})==len(train)+len(val)
    assert len({r['cluster_sha256'] for r in train+val})==len(train)+len(val)
    total=sum(totals.values())
    assert abs(total-target)<=4096*5 and len(train)<=100000
    for b,n in totals.items():assert abs(n/total-plan['bucket_targets'][b]/target)<=.02
    assert flags['contest']/total<=.05 and flags['easy_math']/total<=.05
    assert all(r['complete'] and r['tokens']<=4096 for r in train+val)
    short=[r for r in train if r['bucket']=='short_function']
    assert sum(r['supervised'] for r in short if r['python'])/totals['short_function']>=.8
    write(root/'selection3_status.json',dict(phase='materializing_selected_rows',train_rows=len(train),val_rows=len(val),tokens=dict(totals),optimizer_steps=0,time=time.time()))
    # Seen history is explicitly reported; old training prompt reuse is allowed.
    selected_hashes={r['prompt_sha256'] for r in train+val};seen=set()
    history=sqlite3.connect('file:'+str(storage/'history_prompts.sqlite')+'?mode=ro',uri=True)
    for (h,) in history.execute("SELECT sha FROM memberships NOT INDEXED WHERE kind='train'"):
        if h in selected_hashes:seen.add(h)
    history.close()
    from transformers import AutoTokenizer
    from datasets import Dataset,Features,Sequence,Value
    tok=AutoTokenizer.from_pretrained('/workspace/dflash2_work/models/Qwen3.8-27B',trust_remote_code=True)
    close=tok.encode('</think>',add_special_tokens=False);assert len(close)==1
    handles={};lengths=collections.defaultdict(lambda:collections.defaultdict(list))
    for split,rows in [('train',train),('val',val)]:
        records=[]
        with (out/(split+'_manifest.jsonl')).open('x') as mf:
          for i,r in enumerate(rows):
            p=r['path']
            if p not in handles:handles[p]=open(p,'rb')
            f=handles[p];f.seek(r['offset']);raw=f.read(r['bytes']);assert hashlib.sha256(raw).hexdigest()==r['row_sha256']
            z=json.loads(raw);ids=z['input_ids'];mask=[int(v) for v in z['loss_mask']];pt=z['metadata']['usage']['prompt_tokens']
            expected=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')
            if hasattr(expected,'get'):expected=expected['input_ids']
            assert ids[:pt]==expected and len(expected)==pt
            assert len(ids)==r['tokens']<=4096 and sum(mask)==r['supervised']
            assert sha(json.dumps([ids,z['loss_mask']],separators=(',',':')))==r['input_mask_sha256']
            end=ids.index(close[0],pt);assert end<len(ids)-1
            segments=dict(input_tokens=len(ids),assistant_tokens=sum(mask),thinking_tokens=end-pt+1,final_tokens_including_end=len(ids)-end-1)
            assert segments['thinking_tokens']+segments['final_tokens_including_end']==sum(mask)
            for name,v in segments.items():lengths[split+'/'+r['bucket']][name].append(v)
            manifest={**r,**segments,'prepared_index':i,'split':split,'seen_in_historical_training':r['prompt_sha256'] in seen,
                'prepared_input_mask_sha256':sha(json.dumps([ids,mask],separators=(',',':')))}
            mf.write(json.dumps(manifest,ensure_ascii=True)+'\n')
            records.append(dict(input_ids=ids,loss_mask=mask,length=len(ids)))
        features=Features({'input_ids':Sequence(Value('int64')),'loss_mask':Sequence(Value('int8')),'length':Value('int64')})
        Dataset.from_list(records,features=features).save_to_disk(str(out/('prepared_'+split)))
    for f in handles.values():f.close()
    # Native packing, two passes maximum, equal schedule in both arms.
    sys.path[:0]=[str(root/'source_integration_v2/src'),str(root/'reference')]
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
    def native_packs(rows,seed,epoch):
        result=[]
        for rank in range(2):
            s=MultipackDistributedBatchSamplerV2(8192,[r['tokens'] for r in rows],2,rank,truncate_long_samples=False,seed=seed)
            s.set_epoch(epoch);result.append([[int(x) for x in p] for p in s])
        assert len(result[0])==len(result[1])
        assert all(p and sum(rows[i]['tokens'] for i in p)<=8192 for rr in result for p in rr)
        return result
    packs=[[],[]];updates=[];epoch_counts=[];dropped=[];cap=plan['scope']['input_token_cap_per_arm'];consumed=0
    for epoch in range(2):
        rr=native_packs(train,2026100902,epoch);included=set();before=len(updates)
        usable=len(rr[0])//4*4
        for offset in range(0,usable,4):
            n=sum(train[i]['tokens'] for rank in range(2) for p in rr[rank][offset:offset+4] for i in p)
            if consumed+n>cap:break
            for rank in range(2):packs[rank].extend(rr[rank][offset:offset+4]);included.update(i for p in rr[rank][offset:offset+4] for i in p)
            consumed+=n;updates.append(n)
        dropped.append(dict(epoch=epoch,indices=sorted(set(range(len(train)))-included)))
        epoch_counts.append(len(updates)-before)
    visits=collections.Counter(i for rr in packs for p in rr for i in p);assert max(visits.values())<=2
    actual=collections.Counter();actual_flags=collections.Counter()
    for i,count in visits.items():
        r=train[i];actual[r['bucket']]+=r['supervised']*count
        for flag in ['contest','easy_math','complete']:actual_flags[flag]+=r['supervised']*count*r[flag]
    actual_total=sum(actual.values())
    for b,n in actual.items():assert abs(n/actual_total-plan['bucket_targets'][b]/target)<=.02
    for flag in ['contest','easy_math']:assert actual_flags[flag]/actual_total<=.05
    valpacks=native_packs(val,2026100903,0)
    valcounts=collections.Counter(i for rr in valpacks for p in rr for i in p)
    # Preserve all 1000 validation rows if the native sampler leaves one tail row.
    # Place that row in an existing pack with spare capacity; no training change.
    assert max(valcounts.values())==1
    validation_omitted=sorted(set(range(1000))-set(valcounts))
    val_tail_placements=[]
    for index in validation_omitted:
        done=False
        for rank in range(2):
            for micro in range(len(valpacks[rank])-1,-1,-1):
                pack=valpacks[rank][micro]
                if sum(val[i]['tokens'] for i in pack)+val[index]['tokens']<=8192:
                    pack.append(index);val_tail_placements.append(dict(index=index,rank=rank,microbatch=micro));done=True;break
            if done:break
        assert done,'No spare native validation pack for omitted tail; do not drop validation row'
    assert sorted(i for rr in valpacks for p in rr for i in p)==list(range(1000))
    steps=len(updates);assert steps>20
    schedule=dict(world_size=2,accumulation=4,packed_budget=8192,single_sequence_cap=4096,
        train_packs={str(i):p for i,p in enumerate(packs)},val_packs={str(i):p for i,p in enumerate(valpacks)},
        global_input_tokens_per_update=updates,total_updates=steps,warmup_updates=max(1,int(.05*steps)),
        checkpoint_updates=sorted({math.ceil(steps*x/4) for x in range(1,5)}),actual_input_tokens=consumed,
        actual_supervised_tokens=actual_total,supervised_by_bucket=dict(actual),repeat_counts=dict(collections.Counter(visits.values())),
        updates_by_epoch=epoch_counts,omitted_train_indices_by_epoch=dropped,offline_val_omitted_indices=[],native_val_tail_placements=val_tail_placements,
        train_manifest_sha256=digest(out/'train_manifest.jsonl'),val_manifest_sha256=digest(out/'val_manifest.jsonl'),
        sampling='Exact isolated pinned native MultipackDistributedBatchSamplerV2; only complete 4-microbatch windows, maximum two epochs')
    write(out/'native_schedule.json',schedule)
    stats={b:{k:dict(n=len(v),mean=float(np.mean(v)),p50=float(np.quantile(v,.5)),p90=float(np.quantile(v,.9)),max=max(v),sum=sum(v)) for k,v in ls.items()} for b,ls in lengths.items()}
    mix=dict(pool_rows=len(train),offline_validation_rows=len(val),pool_supervised_tokens=total,pool_buckets=dict(totals),pool_sources=dict(source_totals),
        pool_flags=dict(flags),actual_consumed_buckets=dict(actual),actual_consumed_flags=dict(actual_flags),actual_input_tokens=consumed,
        historical_seen_train_prompts=sum(r['prompt_sha256'] in seen for r in train),historical_seen_val_prompts=sum(r['prompt_sha256'] in seen for r in val),
        complete_fraction=1.,length_statistics=stats,limitations=['All complete 4K rows: shared completeness/length selection bias','1000 offline validation is held out from this new trial, not claimed unseen by historical training','Code-token length requires re-encoding spans and is not equated to exact returned-token segments'])
    write(out/'data_mix_actual.json',mix)
    write(out/'decontamination_report.json',dict(cluster_sha256=report['candidate_sha256'],one_representative_per_cluster=True,
        train_val_cluster_intersection=0,train_val_exact_prompt_intersection=0,all_old_eval_and_confirmation_exclusions_applied=True,
        receipts={n:digest(root/n) for n in ['pool_exclusions_report.json','confirmation_seal_receipt.json','alternative_candidates_v2_report.json','completed_generations_audit.json']},
        limitations=report['limitations']))
    write(out/'taxonomy_audit.json',dict(version=4,manual_200_sha256=digest(root/'classification_manual_review_200_v1.json'),
        new_generation_40_sha256=digest(root/'generated_candidates_manual_review40.json'),census_sha256=digest(root/'taxonomy_v4_report.json')))
    files={str(p):digest(p) for p in out.rglob('*') if p.is_file()}
    files[str(root/'PLAN_DATA_SCALE_REVISION_v3.json')]=digest(root/'PLAN_DATA_SCALE_REVISION_v3.json')
    lock=dict(passed=True,phase='data_locked_gpu_smoke_and_full_trainer_still_required',version=3,time=time.time(),
        train_rows=len(train),offline_val_rows=1000,total_updates=steps,actual_input_tokens=consumed,pool_supervised_tokens=total,
        schedule=str(out/'native_schedule.json'),schedule_sha256=digest(out/'native_schedule.json'),files=files,
        all_four_ranks_same_schedule=True,optimizer_steps=0,main_training_started=False,duration_s=time.time()-began)
    write(out/'SELECTION_LOCK.json',lock);write(root/'selection3_lock_receipt.json',lock);write(root/'selection3_status.json',lock)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--storage',type=Path,required=True);a=p.parse_args()
    try:main(a.run,a.storage)
    except BaseException as e:write(a.run/'selection3_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
