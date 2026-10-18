"""Conservative candidate taxonomy, preserving the immutable intake catalogue.

This is a feasibility census, NOT a selected training dataset. Every label still
requires the predeclared full-row audit before the selection lock.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import traceback
from prepare_inventory import identity, write

FUNCTION = re.compile(r'\b(?:function|method|subroutine)\b|def\s+\w+\s*\(|函数|方法实现', re.I)
CODE = re.compile(r'```|\b(?:python|javascript|typescript|java|c\+\+|c#|programming|algorithm|code|sql|program|implementation)\b|编程|代码', re.I)
REPAIR = re.compile(r'\b(?:debug|fix|repair|refactor|code review|unit tests?)\b|\bexplain\b.{0,100}\b(?:code|function|program|implementation)\b|\b(?:code|function|program)\b.{0,70}\b(?:bug|incorrect|error|modify|change|optimi[sz]e)\b|修复|调试|解释.{0,15}代码', re.I | re.S)
STDIO = re.compile(r'\b(?:standard input|standard output|input specification|output specification|time limit|memory limit|codeforces|codechef|atcoder|stdin|stdout)\b|(?:^|\n)\s*(?:#+\s*)?(?:input|output)\s*(?:format|:)|输入格式|输出格式', re.I)
MATH_TASK = re.compile(r'(?:solve (?:the following |this )?math|math(?:ematical)? problem|solve the problem|find the (?:value|number)|calculate|compute|determine|prove|evaluate|simplify|equation|triangle|polynomial|integral|probability|degrees|how many|how much)', re.I)
EASY = re.compile(r'\b(?:dollars?|apples?|oranges?|cents?|cand(?:y|ies)|marbles?|how many|how much)\b', re.I)
GENERAL = {'ultrachat', 'ultrafeedback', 'autoif', 'lmsys'}
MATH_SOURCE = {'metamathqa', 'metamath', 'orca', 'v2math'}
SHORT_SOURCE = {'evol', 'codefn'}
REPAIR_SOURCE = {'evol', 'codefn', 'ultrainteract', 'rstar'}


def classify(q, source, flags, complete):
    # Tool availability is not a request to implement a Python function.
    body = re.sub(r'^\s*Tool available[^\n]*(?:\n|$)', '', q, flags=re.I)
    if source in MATH_SOURCE:
        return 'math', 'math_source', False
    if source == 'ultrainteract' and MATH_TASK.search(body) and not STDIO.search(body):
        explicit_code = bool(REPAIR.search(body) or re.search(r'\b(?:write|implement|create|modify)\b.{0,100}\b(?:code|function|program|method)\b', body, re.I | re.S))
        if not explicit_code:
            return 'math', 'ultrainteract_math_task_not_tool_language', False
    contest = bool(STDIO.search(body))
    code = bool(CODE.search(body))
    if source in REPAIR_SOURCE and contest:
        return 'code_reasoning_and_repair', 'explicit_stdio_or_contest', True
    if source in REPAIR_SOURCE and code and REPAIR.search(body):
        return 'code_reasoning_and_repair', 'explicit_repair_or_code_explanation', False
    if source in SHORT_SOURCE and code and FUNCTION.search(body):
        if not complete:
            return 'excluded_short_function_truncated', 'primary_code_implementation_must_be_uncut', False
        if not flags['code']['python_function_ast_ok']:
            return 'review_other_language_function', 'no_verified_complete_python_function', False
        return 'short_function', 'allowed_source_function_request_and_python_ast', False
    if source in GENERAL and not (code and (FUNCTION.search(body) or contest or REPAIR.search(body))):
        return 'general_instruction', 'allowed_general_source', False
    return 'review_ambiguous', 'not_confidently_classified', contest


def main(root, storage):
    started = time.time()
    write(root/'taxonomy_v2_identity.json', identity())
    outpath = storage/'taxonomy_v2.sqlite'
    assert not outpath.exists(), 'No fresh overwrite'
    src = sqlite3.connect('file:'+str(storage/'pool_catalogue.sqlite')+'?mode=ro', uri=True)
    src.execute('PRAGMA cache_size=-1048576')
    dst = sqlite3.connect(outpath)
    dst.execute('PRAGMA journal_mode=WAL'); dst.execute('PRAGMA cache_size=-262144')
    dst.execute('CREATE TABLE candidates(pool TEXT,id TEXT,prompt_sha TEXT,source TEXT,bucket TEXT,reason TEXT,complete INT,supervised INT,tokens INT,python INT,contest INT,easy_math INT,sort_sha TEXT,PRIMARY KEY(pool,id))')
    stats = collections.Counter(); buckets = collections.defaultdict(collections.Counter)
    sources = collections.Counter(); reviews = collections.defaultdict(list)
    query = 'SELECT r.pool,r.id,r.prompt_sha,r.source,r.complete,r.supervised,r.tokens,r.flags,p.prompt FROM responses r JOIN prompts p ON r.pool=p.pool AND r.id=p.id WHERE r.eligible=1 ORDER BY r.pool,r.id'
    seen = set()
    for pool,key,h,source,complete,sup,tokens,raw_flags,q in src.execute(query):
        stats['eligible_rows'] += 1
        if h in seen:
            stats['exact_prompt_duplicates_skipped'] += 1
            continue
        seen.add(h); flags=json.loads(raw_flags)
        bucket,reason,contest=classify(q,source,flags,bool(complete))
        python=bool(flags['code']['python_function_ast_ok'])
        easy=bool(EASY.search(q)) if bucket=='math' else False
        order=hashlib.sha256(('2026100901/'+pool+'/'+key+'/'+h).encode()).hexdigest()
        dst.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', (pool,key,h,source,bucket,reason,complete,sup,tokens,int(python),int(contest),int(easy),order))
        b=buckets[bucket];b['rows']+=1;b['supervised_tokens']+=sup;b['complete_rows']+=complete;b['complete_supervised_tokens']+=sup*complete;b['python_complete_supervised_tokens']+=sup*complete*python;b['contest_supervised_tokens']+=sup*contest;b['easy_math_supervised_tokens']+=sup*easy
        sources[source]+=1
        # Deterministic review candidates only. Do not mark these as reviewed.
        v=reviews[bucket];v.append(dict(order_sha256=order,pool=pool,id=key,prompt_sha256=h,source=source,bucket=bucket,reason=reason,prompt=q,complete=bool(complete),supervised=sup,flags=flags))
        if len(v)>96:v.sort(key=lambda x:x['order_sha256']);del v[64:]
        if stats['eligible_rows']%10000==0:
            if time.time()-started>7200:raise TimeoutError('Taxonomy census 2h cap')
            dst.commit();write(root/'taxonomy_v2_status.json',dict(stage='candidate_census',counters=dict(stats),time=time.time(),optimizer_steps=0))
    dst.execute('CREATE INDEX candidate_bucket ON candidates(bucket,complete,sort_sha)');dst.commit();dst.execute('PRAGMA wal_checkpoint(TRUNCATE)');dst.close();src.close()
    for bucket,rows in reviews.items():
        reviews[bucket]=sorted(rows,key=lambda x:x['order_sha256'])[:40]
    write(root/'taxonomy_v2_review_candidates.json',dict(status='NOT_YET_REVIEWED',rows=dict(reviews)))
    report=dict(status='census_complete_not_selection',counters=dict(stats),buckets=dict(buckets),source_unique_counts=dict(sources),duration_s=time.time()-started,remaining=['Full-row taxonomy review including final code','Historical/evaluation/confirmation and near duplicate exclusion','Replay mapping and offline validation','Exact target token mixture and selection lock'])
    write(root/'taxonomy_v2_report.json',report);write(root/'taxonomy_v2_status.json',report)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);x=a.parse_args()
    try:main(x.run,x.storage)
    except BaseException as e:
        write(x.run/'taxonomy_v2_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
