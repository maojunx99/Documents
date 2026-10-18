# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Preselection taxonomy correction after the frozen 200-row category audit.

No model outcomes or benchmark answers enter classification. Original catalogue,
prior censuses and raw responses are read-only. This is NOT a selection lock.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import traceback
import prepare_taxonomy_v2 as base
import prepare_taxonomy_v3 as old

CODE = re.compile(r'```|\b(?:python|javascript|typescript|java|ruby|luau|lua|csharp|programming|algorithm|code|sql|program|implementation|django|matplotlib|recursion|call.stack|type.hints?|exception|attributeerror)\b|c\+\+|c#|编程|代码', re.I)
REPAIR = re.compile(r'\b(?:debug|fix|repair|refactor|review|erroneous|incorrect|error|bug|unit tests?|traceback|exception|not working|doesn.t work)\b|\b(?:explain|understand|describe|analy[sz]e)\b.{0,160}\b(?:code|function|program|implementation|recursion|stack|type.hint|backend)\b|\b(?:why|how does|how do|how can)\b.{0,160}\b(?:work|behav|function|code|program|hint|backend|error)\b|\b(?:modify|change|adapt|correct|optimi[sz]e)\b|\b(?:using|from)\s+(?:the\s+)?(?:given|following|existing|provided)\s+(?:code|function|program)\b|修复|调试|解释.{0,15}代码', re.I|re.S)
CREATE = re.compile(r'\b(?:write|implement|create|design|develop|build|complete|define|generate)\b.{0,160}\b(?:function|method|class|program|code|algorithm|implementation|subroutine)\b|function signature|def\s+\w+\s*\(|实现.{0,30}(?:函数|方法)', re.I|re.S)
STDIO = re.compile(old.STRONG_STDIO.pattern+r'|\b(?:first|next|second|following)\s+(?:\w+\s+){0,3}line[s]?\s+(?:of (?:the )?input\s+)?(?:contains?|consists?)\b', re.I)
MATH = re.compile(r'\b(?:equations?|triangles?|polynomials?|integrals?|probability|degrees|integers?|fractions?|ratios?|percent|percentage|divisible|squares?|roots?|remainder|sum|product|average|mean|median|area|volume|circumference|perimeter|prove|simplify|evaluate)\b|\\(?:frac|sqrt|sum|int|boxed)\b',re.I)
NUMERIC = re.compile(r'\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|half|third)\b',re.I)
TASK = re.compile(r'\b(?:solve|find|calculate|compute|determine|how many|how much|what is|what are)\b',re.I)
EASY = re.compile(base.EASY.pattern+r'|\b(?:years? old|ages?|hours?|minutes?|days?|weeks?|miles?|kilometers?|kilometres?|pounds?|ounces?|gallons?|liters?|litres?|per day|per hour|costs?|prices?|spends?|spent|earns?|earned|saves?|saved|sold|buys?|bought|sells?|jelly beans?|friends|books?|students?|tickets?|workers?|bottles?|boxes|candles?|flowers?|cookies?|slices?|drinks?|took|takes)\b',re.I)
ORPHAN = re.compile(r'^\s*(?:\d+[.)]\s+(?:a |the )?(?:closing|conclusion|summary)\b|(?:does|did|is|was) the (?:film|movie|article|story)\b)',re.I)
FOREIGN = {'java','javascript','js','typescript','ts','cpp','c++','c','csharp','cs','c#','ruby','rb','go','rust','php','swift','kotlin','scala','lua','luau'}


def classify(q, source, flags, complete):
    body=re.sub(r'^\s*Tool available[^\n]*(?:\n|$)', '',q,flags=re.I)
    implementation=bool(CREATE.search(body))
    code=bool(CODE.search(body) or (source in base.SHORT_SOURCE and implementation))
    repair=code and bool(REPAIR.search(body) or re.search(r'\b(?:why is this|why does this|elucidate|mechanics behind|using the code given|using the function given)\b',body,re.I))
    contract=bool(old.FUNCTION_SIGNATURE.search(body))
    contest=bool(STDIO.search(body) or (old.IO_HEADING.search(body) and not contract))
    if ORPHAN.search(body) and len(body)<400:
        return 'review_contextless','orphan_reference_without_context',contest
    if source in base.MATH_SOURCE and not code:
        return 'math','math_source_without_explicit_code',False
    if source=='ultrainteract' and not code and (MATH.search(body) or (TASK.search(body) and NUMERIC.search(body))):
        return 'math','quantitative_task_word_boundary_guard',False
    if source in base.REPAIR_SOURCE and contest:
        return 'code_reasoning_and_repair','explicit_stdio_or_contest',True
    if source in base.REPAIR_SOURCE and repair:
        return 'code_reasoning_and_repair','prompt_explanation_or_modification_before_answer_ast',False
    implicit_contract=bool(re.search(r'\breturn (?:true|false|an?|the)\b',body,re.I) and re.search(r'\bexamples?\b',body,re.I) and re.search(r'\bconstraints\b',body,re.I))
    if source in base.SHORT_SOURCE and (contract or implementation or implicit_contract):
        if not complete:return 'excluded_short_function_truncated','primary_code_implementation_must_be_uncut',False
        if flags['code']['python_function_ast_ok']:
            return 'short_function','explicit_implementation_request_with_complete_python_function',False
        return 'review_other_language_function','no_verified_complete_python_function',False
    if source in base.GENERAL and not code:
        return 'general_instruction','general_source_without_code_task',False
    return 'review_ambiguous','intent_not_sufficiently_confident',contest


def main(root,storage):
    started=time.time();base.write(root/'taxonomy_v4_identity.json',base.identity())
    out=storage/'taxonomy_v4.sqlite';assert not out.exists(),'No fresh overwrite'
    src=sqlite3.connect('file:'+str(storage/'pool_catalogue.sqlite')+'?mode=ro',uri=True)
    dst=sqlite3.connect(out)
    dst.execute('CREATE TABLE candidates(pool TEXT,id TEXT,prompt_sha TEXT,source TEXT,bucket TEXT,reason TEXT,complete INT,supervised INT,tokens INT,python INT,contest INT,easy_math INT,sort_sha TEXT,PRIMARY KEY(pool,id))')
    manual=json.loads((root/'classification_manual_review_200_v1.json').read_text())
    reviewed={(x['pool'],x['id']):x for x in manual['records']}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/workspace/dflash2_work/models/Qwen3.8-27B',trust_remote_code=True)
    counts=collections.Counter();buckets=collections.defaultdict(collections.Counter);reviews=collections.defaultdict(list);seen=set();handles={}
    query='SELECT r.pool,r.id,r.prompt_sha,r.source,r.complete,r.supervised,r.tokens,r.flags,p.prompt,r.path,r.offset,r.bytes,r.row_sha FROM responses r JOIN prompts p ON r.pool=p.pool AND r.id=p.id WHERE r.eligible=1 ORDER BY r.pool,r.id'
    for pool,key,h,source,complete,sup,tokens,rawflags,q,path,offset,size,rowsha in src.execute(query):
        counts['eligible_rows']+=1
        if h in seen:counts['exact_prompt_duplicates_skipped']+=1;continue
        seen.add(h);flags=json.loads(rawflags);bucket,reason,contest=classify(q,source,flags,bool(complete))
        audit=reviewed.get((pool,key))
        if audit:
            decision=audit['decision']
            if decision.startswith(('exclude_','quarantine_')):bucket='review_manual_quarantine';reason=decision
            elif decision=='reclassify_code_reasoning_and_repair':bucket='code_reasoning_and_repair';reason='manual_prompt_intent';contest=False
            elif decision=='reclassify_general_instruction':bucket='general_instruction';reason='manual_nonmath_reading_comprehension';contest=False
            elif decision=='flag_algorithm_contest_style':bucket='code_reasoning_and_repair';reason='manual_stdio_contract';contest=True
        python=False
        if bucket=='short_function':
            f=handles.setdefault(path,open(path,'rb')) if path not in handles else handles[path]
            f.seek(offset);raw=f.read(size);assert hashlib.sha256(raw).hexdigest()==rowsha
            row=json.loads(raw);ids=row['input_ids'];text=tok.decode(ids[flags['prompt_tokens']:],skip_special_tokens=False)
            assert '</think>' in text
            final=text.split('</think>',1)[1].replace('<|im_end|>','').strip()
            blocks=re.findall(r'```([^\n]*)\n(.*?)```',final,re.S)
            foreign=any(lang.strip().lower() in FOREIGN for lang,_ in blocks)
            syntax=True;found=False
            for lang,body in blocks:
                if lang.strip().lower() in {'python','py'}:
                    try:
                        tree=ast.parse(body);found |= any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree))
                    except (SyntaxError,ValueError,RecursionError):syntax=False
            counts['short_raw_rows_rechecked']+=1
            if foreign:bucket='review_mixed_language_function';reason='not_counting_mixed_answer_as_all_python'
            elif not syntax or not found:bucket='review_python_structure';reason='python_syntax_or_primary_function_missing'
            else:python=True
        easy=bool(EASY.search(q)) if bucket=='math' else False
        order=hashlib.sha256(('2026100901/'+pool+'/'+key+'/'+h).encode()).hexdigest()
        dst.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(pool,key,h,source,bucket,reason,complete,sup,tokens,int(python),int(contest),int(easy),order))
        b=buckets[bucket];b['rows']+=1;b['supervised_tokens']+=sup;b['complete_supervised_tokens']+=sup*complete;b['python_complete_supervised_tokens']+=sup*complete*python;b['contest_supervised_tokens']+=sup*contest;b['easy_math_supervised_tokens']+=sup*easy
        rv=reviews[bucket];rv.append(dict(order=order,pool=pool,id=key,prompt=q,bucket=bucket,reason=reason,complete=complete,supervised=sup))
        if len(rv)>80:rv.sort(key=lambda x:x['order']);del rv[40:]
        if counts['eligible_rows']%10000==0:
            assert time.time()-started<7200,'2h CPU cap'
            dst.commit();base.write(root/'taxonomy_v4_status.json',dict(stage='census',counts=dict(counts),time=time.time(),optimizer_steps=0))
    for f in handles.values():f.close()
    dst.execute('CREATE INDEX candidate_bucket ON candidates(bucket,complete,sort_sha)');dst.commit();dst.close();src.close()
    report=dict(status='census_complete_not_selection',counts=dict(counts),buckets=dict(buckets),duration_s=time.time()-started,manual_review_sha256=hashlib.sha256((root/'classification_manual_review_200_v1.json').read_bytes()).hexdigest(),limitations=['Rule labels remain imperfect; not semantic truth','Mixed-language short answers quarantined, not counted as Python','No split or quota lock; decontamination and offline split still required','Simple word-problem flag conservative heuristic; final quota audit required'],optimizer_steps=0)
    base.write(root/'taxonomy_v4_report.json',report);base.write(root/'taxonomy_v4_status.json',report)
    base.write(root/'taxonomy_v4_review_candidates.json',{k:sorted(v,key=lambda x:x['order'])[:40] for k,v in reviews.items()})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);x=a.parse_args()
    try:main(x.run,x.storage)
    except BaseException as e:
        base.write(x.run/'taxonomy_v4_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=base.identity()));raise
