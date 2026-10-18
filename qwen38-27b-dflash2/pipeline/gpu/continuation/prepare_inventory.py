# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Bounded, CPU-only intake. Never launches a teacher, trainer or evaluator.

The catalogue is evidence for selection, not a selected training dataset.
All source files are opened read-only. Output lives in an exclusive attempt.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import traceback
import unicodedata

BASE = Path('/workspace')
W = BASE / 'dflash2_work'


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).split())


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temp.replace(path)


def identity():
    p = Path('/proc/self')
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), starttime=fields[19], pgid=os.getpgid(0),
                cmdline=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode())


CONTEST = re.compile(r'(?:standard input|standard output|input specification|output specification|time limit|memory limit|codeforces|codechef|atcoder|stdin|stdout|test cases?\b)', re.I)
REPAIR = re.compile(r'\b(?:debug|fix|repair|refactor|explain|unit tests?|test cases?|code review)\b', re.I)
FUNCTION = re.compile(r'(?:\b(?:function|method)\b|def\s+\w+\s*\(|function\s+\w+\s*\(|\b(?:implement|write|create)\b.{0,70}\b(?:python|code)\b)', re.I | re.S)
CODE = re.compile(r'```|\b(?:python|javascript|java|c\+\+|programming|algorithm|code|function|method|sql|program)\b', re.I)
EASY_MATH = re.compile(r'\b(?:dollars?|apples?|oranges?|cents?|cand(?:y|ies)|marbles?|how many|how much)\b', re.I)
BAD_FOLLOWUP = re.compile(r'^(?:continue\.?|go on\.?|yes\.?|no\.?|thanks\.?|thank you\.?|explain (?:it|that)|do (?:it|that) again)\s*$', re.I)


def classify(prompt, source):
    # These are conservative, reviewable candidate labels, never semantic truth.
    contest = bool(CONTEST.search(prompt))
    if CODE.search(prompt) and (source in {'evol', 'codefn', 'rstar', 'ultrainteract'} or FUNCTION.search(prompt)):
        if contest or REPAIR.search(prompt):
            bucket = 'code_reasoning_and_repair'
        elif FUNCTION.search(prompt):
            bucket = 'short_function'
        else:
            bucket = 'ambiguous_code'
    elif source in {'metamathqa', 'orca', 'v2math'}:
        bucket = 'math'
    elif source == 'ultrainteract':
        bucket = 'ambiguous_reasoning'
    elif source in {'ultrachat', 'ultrafeedback', 'autoif', 'lmsys'}:
        bucket = 'general_instruction'
    else:
        bucket = 'ambiguous'
    return bucket, contest, bool(EASY_MATH.search(prompt)), bool(re.search(r'\bpython\b|def\s+\w+\(', prompt, re.I))


def code_check(final):
    blocks = re.findall(r'```([^\n]*)\n(.*?)```', final, re.S)
    python_blocks = [body for lang, body in blocks if lang.strip().lower() in {'python', 'py'}]
    parsed = []
    for body in python_blocks:
        try:
            tree = ast.parse(body)
            parsed.append(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)))
        except (SyntaxError, ValueError, RecursionError):
            parsed.append(False)
    return dict(code_fences=len(blocks), python_blocks=len(python_blocks),
                python_function_ast_ok=any(parsed), all_python_blocks_parse=bool(parsed) and all(parsed),
                odd_fences=final.count('```') % 2 != 0)


def main(root, storage, hours):
    began = time.time()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    write(root/'preparation_identity.json', identity())
    status = dict(stage='loading_tokenizer', optimizer_steps=0, gpu_processes_launched=0,
                  start_time=began, deadline=began+hours*3600)
    write(root/'preparation_status.json', status)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(W/'models/Qwen3.8-27B'), trust_remote_code=True)
    dbpath = storage/'pool_catalogue.sqlite'
    if dbpath.exists():
        raise RuntimeError('Fresh intake refuses an existing catalogue; inspect status first')
    db = sqlite3.connect(dbpath)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE prompts(pool TEXT,id TEXT,source TEXT,prompt TEXT,prompt_sha TEXT,one_user INT,PRIMARY KEY(pool,id))')
    db.execute('CREATE INDEX prompt_hash ON prompts(prompt_sha)')
    db.execute('CREATE TABLE responses(pool TEXT,id TEXT,path TEXT,offset INT,bytes INT,row_sha TEXT,prompt_sha TEXT,source TEXT,bucket TEXT,tokens INT,supervised INT,complete INT,eligible INT,flags TEXT,PRIMARY KEY(pool,id))')
    files=[]; stats=collections.Counter(); buckets=collections.defaultdict(collections.Counter)

    def tick(stage, **extra):
        if time.time() > status['deadline']:
            raise TimeoutError('Predeclared CPU intake time limit exceeded; no automatic retry')
        status.update(stage=stage, updated=time.time(), counters=dict(stats), **extra)
        write(root/'preparation_status.json', status)

    for pool in ['pen0_500k', 'pen0_supplement']:
        shardroot = BASE/pool/'shards'
        for path in sorted(shardroot.glob('*.jsonl')):
            if path.name.endswith(('_out.jsonl', '.errors.jsonl')):
                continue
            h=hashlib.sha256(); count=0; initial=path.stat()
            with path.open('rb') as f:
                for line in f:
                    h.update(line)
                    if not line.strip(): continue
                    r=json.loads(line); turns=r['conversations']
                    q='\n'.join(t['value'] for t in turns if t.get('from')=='human')
                    one=len(turns)==1 and turns[0].get('from')=='human'
                    db.execute('INSERT INTO prompts VALUES(?,?,?,?,?,?)',(pool,str(r['id']),r['src'],q,sha(norm(q)),int(one)))
                    count+=1; stats['input_rows']+=1
            assert path.stat().st_size==initial.st_size and path.stat().st_mtime_ns==initial.st_mtime_ns
            files.append(dict(path=str(path),sha256=h.hexdigest(),bytes=initial.st_size,mtime_ns=initial.st_mtime_ns,rows=count,role='input'))
        db.commit();tick('indexing_inputs', pool=pool)

    for pool in ['pen0_500k', 'pen0_supplement']:
        inputs={r[0]:r[1:] for r in db.execute('SELECT id,source,prompt,prompt_sha,one_user FROM prompts WHERE pool=?',(pool,))}
        for path in sorted((BASE/pool/'shards').glob('*_out.jsonl')):
            h=hashlib.sha256(); count=0; initial=path.stat()
            with path.open('rb') as f:
                offset=0
                for line in f:
                    start=offset;offset+=len(line);h.update(line)
                    if not line.strip():continue
                    r=json.loads(line); key=str(r['primary_id'])
                    src,prompt,prompt_sha,one=inputs[key]
                    ids=r['input_ids'];mask=r['loss_mask'];m=r['metadata'];sp=m.get('sampling_params',{})
                    bad=[]
                    if len(ids)!=len(mask) or not all(v in (0,1) for v in mask):bad.append('invalid_mask')
                    if any(sp.get(k)!=v for k,v in dict(temperature=1.0,top_p=.95,top_k=20,presence_penalty=0.0,reasoning_effort='xhigh').items()):bad.append('sampling_mismatch')
                    if not one:bad.append('not_single_user')
                    expected=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')
                    if hasattr(expected,'get'):expected=expected['input_ids']
                    pt=m.get('usage',{}).get('prompt_tokens')
                    if ids[:len(expected)]!=expected or pt!=len(expected):bad.append('native_prompt_token_mismatch')
                    if mask[:len(expected)]!=[0]*len(expected) or not all(mask[len(expected):]):bad.append('assistant_mask_mismatch')
                    if not prompt.strip() or BAD_FOLLOWUP.fullmatch(prompt.strip()):bad.append('empty_or_contextless')
                    text=r.get('text',''); response=tokenizer.decode(ids[len(expected):],skip_special_tokens=False)
                    final=response.split('</think>',1)[1] if '</think>' in response else ''
                    final=final.replace('<|im_end|>','').strip()
                    cs=code_check(final);bucket,contest,easy,python=classify(prompt,src)
                    sup=sum(mask[:4096]);complete=m.get('finish_reason')=='stop' and len(ids)<=4096 and bool(final)
                    if sup<=8:bad.append('insufficient_supervision')
                    if complete and cs['odd_fences']:bad.append('unclosed_code_fence')
                    if bucket=='short_function' and complete and (python or cs['python_blocks']) and not cs['python_function_ast_ok']:
                        bad.append('no_parseable_python_function')
                    flags=dict(bad=bad,finish_reason=m.get('finish_reason'),prompt_tokens=len(expected),raw_tokens=len(ids),
                               raw_supervised=sum(mask),has_think_end='</think>' in response,nonempty_final=bool(final),
                               contest=contest,easy_math=easy,python=python or cs['python_function_ast_ok'],code=cs,
                               taxonomy='deterministic_candidate_label_requires_review',
                               input_ids_sha256=sha(json.dumps(ids,separators=(',',':'))),
                               loss_mask_sha256=sha(json.dumps(mask,separators=(',',':'))))
                    db.execute('INSERT INTO responses VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (pool,key,str(path),start,len(line),hashlib.sha256(line).hexdigest(),prompt_sha,src,bucket,min(len(ids),4096),sup,int(complete),int(not bad),json.dumps(flags)))
                    stats['output_rows']+=1;stats['eligible_candidate_rows']+=not bad;stats['complete_candidate_rows']+=complete and not bad
                    for issue in bad:stats['rejected_'+issue]+=1
                    b=buckets[bucket];b['rows']+=1
                    if not bad:
                        b['eligible_rows']+=1;b['eligible_supervised_tokens']+=sup
                        b['complete_rows']+=complete;b['complete_supervised_tokens']+=sup*complete
                        b['python_complete_tokens']+=sup*complete*flags['python']
                        b['contest_tokens']+=sup*contest;b['easy_math_tokens']+=sup*easy
                    count+=1
                    if stats['output_rows']%1000==0:db.commit();tick('auditing_all_responses',pool=pool,file=path.name,buckets=dict(buckets))
            assert path.stat().st_size==initial.st_size and path.stat().st_mtime_ns==initial.st_mtime_ns
            files.append(dict(path=str(path),sha256=h.hexdigest(),bytes=initial.st_size,mtime_ns=initial.st_mtime_ns,rows=count,role='output'))
            db.commit()
        del inputs
    duplicates=db.execute('SELECT COUNT(*)-COUNT(DISTINCT prompt_sha) FROM prompts').fetchone()[0]
    db.execute('CREATE INDEX response_bucket ON responses(bucket,eligible,complete)');db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    write(root/'source_fingerprints_intake.json',dict(files=files,tokenizer_path=str(W/'models/Qwen3.8-27B')))
    write(root/'pool_inventory.json',dict(counters=dict(stats),buckets=dict(buckets),input_exact_duplicates=duplicates,
          catalogue=str(dbpath),scope='Full pool quality and exact native prompt audit only; not decontaminated/selected',
          remaining=['Historical and evaluation exclusion index','Near-duplicate clusters','Manual taxonomy audit','Confirmation seal','Final token mixture and schedule','Training implementations and preflights'],
          duration_s=time.time()-began))
    tick('intake_complete_selection_pending',buckets=dict(buckets))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);a.add_argument('--max-hours',type=float,default=2)
    args=a.parse_args()
    try:main(args.run,args.storage,args.max_hours)
    except BaseException as e:
        write(args.run/'preparation_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()))
        raise
