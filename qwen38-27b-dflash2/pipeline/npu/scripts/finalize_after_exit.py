# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Finite CPU follower: verify and archive this run when its supervisor exits."""
import argparse, json, os, pathlib, subprocess, time, traceback

ap=argparse.ArgumentParser();ap.add_argument('--run',type=pathlib.Path,required=True);a=ap.parse_args()
run=a.run.resolve();base=run.parents[1];py=os.environ.get('DFLASH_PYTHON', __import__('sys').executable)
identity=json.loads((run/'launch_identity.json').read_text())
mine={'pid':os.getpid(),'starttime':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
(run/'finalization_identity.json').write_text(json.dumps(mine,indent=2))
deadline=time.time()+22200
try:
    while True:
        p=pathlib.Path(f'/proc/{identity["pid"]}/stat')
        if not p.exists():break
        s=p.read_text().rsplit(')',1)[1].split()
        if s[19]!=str(identity['starttime']) or s[0]=='Z':break
        if time.time()>deadline:raise TimeoutError('Finite finalization deadline')
        time.sleep(5)
    if (run/'failure.json').exists():raise RuntimeError('Native workflow failed; preserving evidence')
    check=subprocess.run([py,str(base/'scripts/verify_completion.py'),'--run',str(run)],capture_output=True,text=True)
    (run/'independent_verification.log').write_text(check.stdout+'\n'+check.stderr)
    if check.returncode:raise RuntimeError('Independent completion verification failed')
    receipt=subprocess.run([py,str(base/'scripts/archive_workflow.py'),'--base',str(base),'--name',run.name+'_completed_workflow'],capture_output=True,text=True)
    if receipt.returncode:raise RuntimeError('Archive verification failed: '+receipt.stderr)
    (run/'FINAL_ARCHIVE_RECEIPT.json').write_text(receipt.stdout)
    print(receipt.stdout,flush=True)
except BaseException:
    (run/'finalization_failure.json').write_text(json.dumps({'traceback':traceback.format_exc(),'utc_epoch':time.time()},indent=2));raise
