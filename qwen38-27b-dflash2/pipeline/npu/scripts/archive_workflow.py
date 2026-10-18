"""Seal scripts/source and lightweight evidence; omit model weights and caches."""
import argparse, hashlib, json, pathlib, tarfile, time

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True,type=pathlib.Path)
    ap.add_argument('--name',required=True);a=ap.parse_args();b=a.base.resolve()
    out=b/'archives';out.mkdir(exist_ok=True)
    package=out/(a.name+'.tar.gz');assert not package.exists()
    files=[]
    for p in b.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(b)
        if any(x in rel.parts for x in ['archives','checkpoints','prepared','hidden_states','__pycache__','torch_compile_debug']):continue
        if p.name=='selected_10000.jsonl' or p.suffix in ['.pyc','.safetensors','.pt','.arrow']:continue
        if p.stat().st_size>30*1024*1024:continue
        files.append(p)
    hashes={str(p.relative_to(b)):digest(p) for p in sorted(files)}
    manifest={'created_utc_epoch':time.time(),'files':hashes,'omissions':'large model/checkpoint/hidden/cache/raw selected dataset; dataset manifests and original paths retained'}
    receipt=out/(a.name+'_MANIFEST.json');receipt.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(package,'w:gz') as t:
        for p in sorted(files):t.add(p,arcname=str(p.relative_to(b)),recursive=False)
        t.add(receipt,arcname='ARCHIVE_MANIFEST.json',recursive=False)
    with tarfile.open(package) as t:
        for name,h in hashes.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h
    result={'archive':str(package),'bytes':package.stat().st_size,'sha256':digest(package),'verified_files':len(files)}
    (out/(a.name+'_RECEIPT.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
