# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Read-only candidate projection and duplicate clusters before any split/lock.

Existing source catalogues stay immutable. This creates a new CFS projection,
not a final training dataset. No model scoring or generated code execution.
"""
import argparse, ast, collections, hashlib, json, re, sqlite3, sys, time, traceback
from pathlib import Path
from prepare_inventory import norm, sha, write, identity
from prepare_confirmation_seal import shingles, signature
from prepare_taxonomy_v4 import STDIO, EASY

BUCKETS=['short_function','code_reasoning_and_repair','general_instruction','math']
CAPS={'short_function':100_000_000,'code_reasoning_and_repair':10_000_000,
      'general_instruction':15_000_000,'math':14_000_000}

def readonly(p):
    return sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)

class CanonicalFunction(ast.NodeTransformer):
    """Alpha-normalize local names only, retaining constants/external APIs."""
    def __init__(self, node):
        names=[node.name]
        names += [x.arg for x in ast.walk(node) if isinstance(x,ast.arg)]
        names += [x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)]
        self.names={v:'local_'+str(i) for i,v in enumerate(dict.fromkeys(names))}
    def visit_Name(self,n):
        n.id=self.names.get(n.id,n.id);return n
    def visit_arg(self,n):
        n.arg=self.names.get(n.arg,n.arg);n.annotation=None;return n
    def visit_FunctionDef(self,n):
        n.name=self.names.get(n.name,n.name);n.returns=None;n.type_comment=None
        n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
        return self.generic_visit(n)
    visit_AsyncFunctionDef=visit_FunctionDef

def primary_function_hash(final):
    for lang,body in re.findall(r'```([^\n]*)\n(.*?)```',final,re.S):
        if lang.strip().lower() not in {'python','py'}:continue
        try:tree=ast.parse(body)
        except (SyntaxError,ValueError,RecursionError):continue
        funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        if funcs:
            n=funcs[0];n=CanonicalFunction(n).visit(n)
            return sha(ast.dump(n,include_attributes=False))
    return None

def main(root,storage):
    began=time.time();deadline=began+7200;write(root/'cluster_candidates_identity.json',identity())
    out=storage/'cluster_candidates_v1';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(root/'deps'))
    from datasketch import MinHashLSH
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/workspace/dflash2_work/models/Qwen3.8-27B',trust_remote_code=True)
    counts=collections.Counter()
    def status(phase,**kw):
        assert time.time()<deadline,'2h preparation cap'
        write(root/'cluster_candidates_status.json',dict(phase=phase,time=time.time(),counts=dict(counts),optimizer_steps=0,**kw))
    audit=json.loads((root/'completed_generations_audit.json').read_text());assert audit['passed']
    generated=storage/'generated_candidates_v1.jsonl'
    assert hashlib.sha256(generated.read_bytes()).hexdigest()==audit['manifest_sha256']
    exclusions=readonly(storage/'pool_exclusions.sqlite')
    excluded={x[0] for x in exclusions.execute('SELECT prompt_sha FROM excluded')};exclusions.close()
    manual=json.loads((root/'generated_candidates_manual_review40.json').read_text())
    rejected={(r['pool'],r['id']) for r in manual['records'] if r['decision'].startswith('quarantine')}
    oldmanual=json.loads((root/'classification_manual_review_200_v1.json').read_text())
    rejected |= {(r['pool'],r['id']) for r in oldmanual['records'] if r['decision'].startswith(('quarantine','exclude'))}
    tax=readonly(storage/'taxonomy_v4.sqlite');tax.row_factory=sqlite3.Row
    bybucket=collections.defaultdict(list)
    for row in tax.execute('SELECT * FROM candidates'):
        r=dict(row);b=r['bucket']
        if b not in BUCKETS or not r['complete'] or r['prompt_sha'] in excluded:continue
        if (r['pool'],r['id']) in rejected:continue
        # Keep contest only in replay for this finite pool; its global cap remains.
        if b=='code_reasoning_and_repair' and r['contest']:continue
        bybucket[b].append(r)
    tax.close();selected={}
    for b,rows in bybucket.items():
        total=0;sources=collections.Counter();easy=0
        for r in sorted(rows,key=lambda x:x['sort_sha']):
            if b=='general_instruction' and sources[r['source']]+r['supervised']>6_000_000:continue
            if b=='math' and r['easy_math'] and easy+r['supervised']>2_000_000:continue
            selected[(r['pool'],r['id'])]=r;total+=r['supervised'];sources[r['source']]+=r['supervised'];easy+=r['supervised']*r['easy_math']
            if total>=CAPS[b]:break
        counts[b+'_staged_tokens']=total
    status('projecting_selected_catalogue_rows',staged=len(selected))
    cat=readonly(storage/'pool_catalogue.sqlite');cat.row_factory=sqlite3.Row
    for rr in cat.execute('SELECT pool,id,path,offset,bytes,row_sha,flags FROM responses'):
        r=selected.get((rr['pool'],rr['id']))
        if r is not None:
            r.update({k:rr[k] for k in ['path','offset','bytes']});r['row_sha256']=rr['row_sha'];r['intake_flags']=json.loads(rr['flags'])
    for rr in cat.execute('SELECT pool,id,prompt FROM prompts'):
        r=selected.get((rr['pool'],rr['id']))
        if r is not None:r['prompt']=rr['prompt']
    cat.close()
    rows=list(selected.values())
    for r in rows:r['prompt_sha256']=r.pop('prompt_sha')
    for line in generated.read_text().splitlines():
        r=json.loads(line)
        if r['eligible'] and (r['pool'],r['id']) not in rejected and r['prompt_sha256'] not in excluded:rows.append(r)
    replay=[json.loads(l) for l in (storage/'replay_index.jsonl').read_text().splitlines()]
    for r in sorted(replay,key=lambda x:sha('2026100901/replay/'+x['primary_id'])):
        if r['split']!='train' or r['prompt_sha256'] in excluded:continue
        if ('replay',r['primary_id']) in rejected:continue
        m=r['metadata'];sup=r['supervised']
        if m.get('finish_reason')!='stop' or m['usage']['total_tokens']>4096:continue
        source='replay_'+r['primary_id'].split('_')[0]
        if counts[source+'_tokens']+sup>1_250_000:continue
        rr={k:r[k] for k in ['path','offset','bytes','row_sha256','prompt','prompt_sha256','tokens','supervised','prepared_indices']}
        rr.update(pool='replay',id=r['primary_id'],source=source,bucket='replay',complete=True,python=False,contest=bool(STDIO.search(r['prompt'])),easy_math=bool(EASY.search(r['prompt'])),sort_sha=sha('2026100901/replay/'+r['primary_id']),reason='original_training_split_only')
        rows.append(rr);counts['replay_staged_tokens']+=sup;counts[source+'_tokens']+=sup
        if counts['replay_staged_tokens']>=5_000_000:break
    # Complete answers, token masks, final hashes and conservative AST equivalence.
    handles={};valid=[]
    with (out/'rejections.jsonl').open('x') as reject:
      for r in sorted(rows,key=lambda x:x['sort_sha']):
        assert 'prompt' in r and 'path' in r
        key=r['pool']+'/'+r['id'];p=r['path']
        if p not in handles:handles[p]=open(p,'rb')
        f=handles[p];f.seek(r['offset']);raw=f.read(r['bytes']);assert hashlib.sha256(raw).hexdigest()==r['row_sha256']
        z=json.loads(raw);ids=z['input_ids'];mask=z['loss_mask'];pt=z['metadata']['usage']['prompt_tokens']
        assert len(ids)==len(mask) and all(v in (0,1) for v in mask)
        assert mask==[0]*pt+[1]*(len(ids)-pt)
        assert sha(norm(r['prompt']))==r['prompt_sha256']
        assert len(ids)<=4096 and z['metadata']['finish_reason']=='stop'
        text=tok.decode(ids[pt:],skip_special_tokens=False)
        if '</think>' not in text or not text.split('</think>',1)[1].replace('<|im_end|>','').strip():
            counts['missing_complete_final']+=1;reject.write(json.dumps(dict(key=key,reason='missing_thinking_closure_or_final'))+'\n');continue
        final=text.split('</think>',1)[1].replace('<|im_end|>','').strip()
        assert len(ids)==r['tokens'] and sum(mask)==r['supervised']
        r.update(key=key,input_mask_sha256=sha(json.dumps([ids,mask],separators=(',',':'))),final_sha256=sha(norm(final)),primary_python_ast_sha256=primary_function_hash(final) if r['bucket']=='short_function' else None,raw_tokens=len(ids))
        r.pop('intake_flags',None);valid.append(r)
        if len(valid)%1000==0:status('checking_raw_candidate_rows',valid=len(valid))
    for f in handles.values():f.close()
    # Union all retrieved duplicates before selecting any validation or training row.
    parent=list(range(len(valid)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    lsh=MinHashLSH(num_perm=128,params=(32,4));sets={};exact={};answers={};functions={};ids={}
    with (out/'cluster_edges.jsonl').open('x') as edges:
      def edge(i,j,kind,sim=None):
        union(i,j);counts['edge_'+kind]+=1;edges.write(json.dumps(dict(a=valid[i]['key'],b=valid[j]['key'],kind=kind,similarity=sim))+'\n')
      for i,r in enumerate(valid):
        ids[r['id']]=i
        for mapping,value,kind in [(exact,r['prompt_sha256'],'exact_prompt'),(answers,r['final_sha256'],'exact_final_answer'),(functions,r['primary_python_ast_sha256'],'alpha_normalized_primary_function')]:
            if value:
                if value in mapping:edge(i,mapping[value],kind)
                else:mapping[value]=i
        s=shingles(r['prompt']);sig=signature(s)
        for j in lsh.query(sig):
            sim=len(s&sets[j])/len(s|sets[j])
            if sim>=.85:edge(i,j,'prompt_5gram_jaccard',sim)
        sets[i]=s;lsh.insert(i,sig)
        if (i+1)%1000==0:status('clustering_before_split',processed=i+1,total=len(valid))
      for a,b in manual['manual_duplicate_pairs']:
        if a in ids and b in ids:edge(ids[a],ids[b],'manual_same_contract')
    groups=collections.defaultdict(list)
    for i,r in enumerate(valid):groups[find(i)].append(i)
    summary=collections.defaultdict(collections.Counter)
    with (out/'candidates.jsonl').open('x') as sink:
      for g,indices in groups.items():
        cluster=sha('\n'.join(sorted(valid[i]['key'] for i in indices)))
        for i in indices:
            r=valid[i];r['cluster_sha256']=cluster;r['cluster_size']=len(indices);sink.write(json.dumps(r,ensure_ascii=True)+'\n')
        # Conservative capacity using one deterministic representative per cluster.
        rep=min((valid[i] for i in indices),key=lambda x:(x['bucket']!='replay',x['sort_sha']))
        summary[rep['bucket']]['clusters']+=1;summary[rep['bucket']]['supervised_tokens']+=rep['supervised']
    report=dict(phase='clustered_candidates_not_selection_lock',valid_rows=len(valid),clusters=len(groups),counts=dict(counts),one_representative_capacity=dict(summary),candidate_sha256=hashlib.sha256((out/'candidates.jsonl').read_bytes()).hexdigest(),duration_s=time.time()-began,optimizer_steps=0,complete_only_projection=True,limitations=['MinHash retrieval not exhaustive semantic duplicate proof','Exact final answers and alpha-normalized primary Python functions conservatively merge some distinct contracts','Completeness is a data intervention; no claim of factual or code correctness','No train/val split or final mixture yet'])
    write(root/'cluster_candidates_report.json',report);write(root/'cluster_candidates_status.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--storage',type=Path,required=True);a=p.parse_args()
    try:main(a.run,a.storage)
    except BaseException as e:write(a.run/'cluster_candidates_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=identity()));raise
