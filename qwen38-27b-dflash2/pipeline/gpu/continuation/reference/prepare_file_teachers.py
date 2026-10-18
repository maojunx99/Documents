# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request
from run_selector_experiment import dump, identity, alive, sha, cleanup_teachers

O=Path(__file__).resolve().parent
W=Path('/workspace/dflash2_work')
PY=str(W/'venv/bin/python')
REPO=Path('/workspace/speculators')
CACHE=Path('/dev/shm/selector_overlap_seed43_20261009_01')


def main():
    assert not (O/'teacher_launch_identity.json').exists()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(l.split(',')[1])<128 for l in gpu.splitlines()),gpu
    import socket
    for port in [8250,8260,8261,8262,8263]:
        with socket.socket() as sock:
            sock.settimeout(1);assert sock.connect_ex(('127.0.0.1',port))!=0,f'Port {port} already occupied'
    dump(O/'teacher_launch_identity.json',identity(os.getpid()))
    CACHE.mkdir(exist_ok=False)
    services=[];processes=[]
    for gpu in range(4):
        d=O/'teachers'/str(gpu);d.mkdir(parents=True)
        cmd=[PY,str(REPO/'scripts/launch_vllm.py'),'train',str(W/'models/Qwen3.8-27B'),
             '--hidden-states-backend','file','--hidden-states-path',str(CACHE),
             '--target-layer-ids','6','20','34','48','62','--include-last-layer',
             '--provenance-dir',str(d/'provenance'),
             '--tensor-parallel-size','1','--max-model-len','16384','--api-server-count','1',
             '--mm-processor-cache-gb','0','--port',str(8260+gpu)]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        log=(d/'server.log').open('wb')
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
        processes.append(p);services.append(identity(p.pid));dump(d/'identity.json',services[-1]);dump(d/'command.json',cmd)
    log=(O/'router.log').open('wb')
    p=subprocess.Popen([PY,str(O/'teacher_dp4_router.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    services.append(identity(p.pid));processes.append(p)
    dump(O/'file_teacher_identities.json',services)
    deadline=time.monotonic()+1800
    while True:
        assert all(p.poll() is None for p in processes),'Teacher process exited'
        ready=[]
        for port in range(8260,8264):
            try:ready.append(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).status==200)
            except Exception:ready.append(False)
        dump(O/'teacher_status.json',{'ready':ready,'time':time.time()})
        if all(ready):break
        assert time.monotonic()<deadline,'Teacher startup timeout'
        time.sleep(5)
    # Refresh cmdline after launcher exec, preserve exact same PID/starttime.
    refreshed=[identity(r['pid']) for r in services]
    assert all(a['starttime']==b['starttime'] for a,b in zip(services,refreshed))
    dump(O/'file_teacher_identities.json',refreshed)
    log=(O/'supervisor.log').open('xb')
    trainer=subprocess.Popen(['python3',str(O/'run_selector_experiment.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=O)
    dump(O/'dispatched_supervisor.json',identity(trainer.pid))


if __name__=='__main__':
    try:main()
    except BaseException:
        dump(O/'teacher_failure.json',{'error':traceback.format_exc(),'time':time.time()})
        raise
