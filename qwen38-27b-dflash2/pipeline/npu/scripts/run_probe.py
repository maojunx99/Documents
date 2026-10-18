# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Finite supervisor for one unique four-NPU FSDP preflight."""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument('--run', type=pathlib.Path, required=True)
a = ap.parse_args()
run = a.run.resolve()
run.mkdir(parents=True, exist_ok=False)
base = pathlib.Path(__file__).resolve().parents[1]
py = os.environ.get('DFLASH_PYTHON', __import__('sys').executable)


def identity(pid):
    try:
        p = pathlib.Path(f'/proc/{pid}')
        s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'starttime': s[19], 'pgid': int(s[2]), 'state': s[0],
                'cmdline': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


(run / 'launch_identity.json').write_text(json.dumps(identity(os.getpid()), indent=2))
env = os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES=os.environ.get('DFLASH_TRAIN_DEVICES','8,9,10,11'), TORCHDYNAMO_DISABLE='1', TASK_QUEUE_ENABLE='1',
           OMP_NUM_THREADS='4', HCCL_EXEC_TIMEOUT='300', PREFLIGHT_OUT=str(run / 'audit'),
           PYTHONPATH=str(base / 'scripts') + ':/usr/local/Ascend/cann-9.1.0/python/site-packages:' + env.get('PYTHONPATH', ''))
argv = [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4', str(base / 'scripts/preflight_fsdp.py')]
p = None
try:
    (run / 'status.json').write_text(json.dumps({'stage': 'preflight_running', 'utc_epoch': time.time()}))
    with (run / 'process.log').open('x') as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        ident = identity(p.pid)
        (run / 'worker_identity.json').write_text(json.dumps(ident, indent=2))
        p.wait(timeout=600)
        assert p.returncode == 0, p.returncode
    assert all(json.loads((run / f'audit/passed_rank{i}.json').read_text())['passed'] for i in range(4))
    (run / 'status.json').write_text(json.dumps({'stage': 'passed', 'utc_epoch': time.time()}))
except BaseException:
    (run / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc(), 'utc_epoch': time.time()}, indent=2))
    raise
finally:
    if p is not None and p.poll() is None:
        now = identity(p.pid)
        if now and now['starttime'] == ident['starttime'] and now['pgid'] == ident['pgid']:
            os.killpg(ident['pgid'], signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                now = identity(p.pid)
                if now and now['starttime'] == ident['starttime']:
                    os.killpg(ident['pgid'], signal.SIGKILL)
                p.wait(timeout=15)
    (run / 'cleanup.json').write_text(json.dumps({'worker': ident if p else None, 'after': identity(p.pid) if p else None}, indent=2))
