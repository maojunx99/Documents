"""Freeze one additional prompt batch within the original aggregate budget."""
import argparse, collections, hashlib, json, re, time
from pathlib import Path
from prepare_inventory import write

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(root, storage):
    review=json.loads((root/'alternative_prompt_review_final.json').read_text())
    assert review['full_prompts_reviewed']==80
    assert review['admitted_source']=='nvidia/OpenCodeInstruct'
    prior=json.loads((root/'generation/completion_verification.json').read_text())
    assert prior['owned_servers_exited'] and prior['target_fingerprints_unchanged']
    result=prior['response_result']; used=result['budget']
    assert used['started']==used['completed']==2685 and not used['unresolved_reserved_tokens']
    source=storage/'alternative_candidates_02/decontaminated_candidates.jsonl'
    report=json.loads((root/'alternative_candidates_v2_report.json').read_text())
    assert digest(source)==report['sha256']
    rejects={x['candidate_id'] for x in review['decisions'] if x['decision']!='accept'}
    counts=collections.Counter(); rows=[]; seen_names=set()
    topics=collections.Counter()
    for row in sorted((json.loads(l) for l in source.read_text().splitlines()),key=lambda r:r['order_sha256']):
        if row['source']!=review['admitted_source']:continue
        if row['candidate_id'] in rejects:counts['manual_reject']+=1;continue
        prompt=row['prompt']
        if re.search(r'\b(undefined|print a single|file_path|symlink)\b',prompt,re.I):counts['extra_quality_rule']+=1;continue
        statuses=json.loads(row['metadata']['tests_execution_status'])
        assert statuses and all(x=='pass' for x in statuses)
        # Limit repeated named contracts and conspicuously repetitive algorithm families.
        names=re.findall(r'\bfunction\s+(?:named\s+|called\s+)?`([a-zA-Z_]\w*)',prompt,re.I)
        if names and names[0].lower() in seen_names:counts['repeated_function_contract']+=1;continue
        topic=next((k for k,rx in [('prime',r'\bprime numbers?\b'),('fibonacci',r'\bfibonacci\b'),('binary_search',r'\bbinary search\b'),('factorial',r'\bfactorial\b')] if re.search(rx,prompt,re.I)),None)
        if topic and topics[topic]>=10:counts['algorithm_family_cap']+=1;continue
        if names:seen_names.add(names[0].lower())
        if topic:topics[topic]+=1
        x={k:row[k] for k in ['prompt','prompt_sha256','revision','prompt_tokens','source']}
        x['id']='opencode_'+row['id'];x['source_candidate_id']=row['candidate_id']
        rows.append(x)
        if len(rows)==1024:break
    assert len(rows)==1024 and len({r['prompt_sha256'] for r in rows})==1024
    child=root/'additional_generation_02'; child.mkdir(exist_ok=False)
    dest=storage/'additional_generation_02';dest.mkdir(exist_ok=False)
    lock=dest/'fallback_generation_prompts_locked.jsonl'
    lock.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
    residual=dict(max_prompts=10000-used['started'],max_completion_tokens=20000000-used['completion_tokens'],absolute_deadline=result['deadline'],previous_used=used,previous_completion_sha256=digest(root/'generation/completion_verification.json'),previous_result_sha256=digest(root/'generation/result.json'),original_limits=dict(prompts=10000,tokens=20000000,wall_seconds=10800),semantics='Same original absolute generation deadline including idle interval; no fresh wall budget')
    assert residual['max_completion_tokens']>8192 and residual['absolute_deadline']>time.time()+1200
    write(child/'RESIDUAL_BUDGET.json',residual)
    write(child/'fallback_prompt_audit_v2.json',dict(passed=True,generation_rows=len(rows),generation_prompts_sha256=digest(lock),source_review_sha256=digest(root/'alternative_prompt_review_final.json'),candidate_report_sha256=digest(root/'alternative_candidates_v2_report.json'),extra_filter_counts=dict(counts),algorithm_family_counts=dict(topics),source_policy='Source prompt only, no external output/test solution',limitations=['40 OpenCode prompts reviewed; remaining prompts not individually semantically certified','First 100K source shard only; lexical decontamination is not semantic proof','Responses require final quality and complete-4K gates']))
    (child/'confirmation_seal_receipt.json').write_bytes((root/'confirmation_seal_receipt.json').read_bytes())
    protocol=dict(phase='additional_target_generation_with_residual_original_budget',source=review['admitted_source'],rows=len(rows),prompt_sha256=digest(lock),sampling=dict(temperature=1.,top_p=.95,top_k=20,presence_penalty=0.,thinking=True,reasoning_effort='xhigh',max_tokens=8192),messages='Original single user only, no manual system or answer hint',budget=residual,zero_optimizer_steps=True,not_training_launch=True,provenance='Direct vLLM argv/env/model/source fingerprints preserved per server',time=time.time())
    write(child/'GENERATION_PROTOCOL.json',protocol)
    write(root/'additional_generation_preparation.json',dict(run=str(child),storage=str(dest),protocol=protocol,time=time.time()))
    print(json.dumps(dict(run=str(child),storage=str(dest),rows=len(rows),residual=residual)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--storage',type=Path,required=True);a=p.parse_args();prepare(a.run,a.storage)
