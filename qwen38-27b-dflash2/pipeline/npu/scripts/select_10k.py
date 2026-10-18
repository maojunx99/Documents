"""Select exactly 10k existing 27B-generated PB records; never re-sample answers."""
import argparse, collections, hashlib, itertools, json, pathlib, random

COUNTS = {'autoif':37389, 'evol_codealpaca':107789, 'lmsys':49002,
          'metamathqa':141738, 'orca_math':183725, 'ultrachat':201774,
          'ultrafeedback':59794, 'ultrainteract':78583}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,required=True)
    ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    quotas={k:int(v/sum(COUNTS.values())*10000) for k,v in COUNTS.items()}
    for k in sorted(COUNTS,key=lambda k:-(COUNTS[k]/sum(COUNTS.values())*10000-quotas[k]))[:10000-sum(quotas.values())]:quotas[k]+=1
    rows=[];seen=set();bad=collections.Counter();selected=[];inputs={}
    for source,quota in quotas.items():
        files=sorted(a.source.glob(source+'_*_out.jsonl'))
        if not files:raise RuntimeError('Missing source '+source)
        streams=[p.open() for p in files];accepted=0
        try:
            for group in itertools.zip_longest(*streams):
                for p,line in zip(files,group):
                    if line is None:continue
                    row=json.loads(line);ids=row.get('input_ids');mask=row.get('loss_mask')
                    if not isinstance(ids,list) or not isinstance(mask,list) or len(ids)!=len(mask) or any(x not in (0,1,False,True) for x in mask):raise RuntimeError('Invalid input/mask in '+str(p))
                    if not any(mask[:4096]):bad['no_supervision_in_first4096']+=1;continue
                    h=hashlib.sha256(json.dumps([ids,mask],separators=(',',':')).encode()).hexdigest()
                    if h in seen:bad['duplicate_sequence']+=1;continue
                    seen.add(h);rows.append(row);accepted+=1
                    selected.append({'source':source,'file':str(p),'id':row.get('id'),'sha256':h,'input_tokens':len(ids),'used_tokens':min(len(ids),4096),'supervised_used':sum(mask[:4096])})
                    inputs[str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                    if accepted==quota:break
                if accepted==quota:break
        finally:
            for stream in streams:stream.close()
        if accepted!=quota:raise RuntimeError(f'{source}: {accepted} != {quota}')
    assert len(rows)==10000 and len(seen)==10000
    order=list(range(10000));random.Random(42).shuffle(order)
    output=a.output/'selected_10000.jsonl'
    with output.open('w') as f:
        for i in order:f.write(json.dumps(rows[i],ensure_ascii=False,separators=(',',':'))+'\n')
    manifest={'rows':10000,'selection_seed':42,'policy':'deterministic interleaving of source shards, approximate original PB source proportions; existing generated token IDs retained; native prepare truncates to4096','source_quotas':quotas,'rejected':dict(bad),'source_files':inputs,'selected_order':[selected[i] for i in order],'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (a.output/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['selected_order','source_files']}),flush=True)

if __name__=='__main__':main()
