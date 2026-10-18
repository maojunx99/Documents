"""Create this bundle's real-device preflight receipts; never reuse historical receipts."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent / 'common'))
from managed import assert_free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-devices', default='8,9,10,11')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    pre = BASE / 'preflight'
    commands = [
        [a.python, str(BASE/'scripts/preflight_checkpointing.py')],
        [a.python, str(BASE/'scripts/preflight_cpu_checkpoint.py')],
        [a.python, str(BASE/'scripts/run_probe.py'), '--run', str(BASE/'probes/toy_01')],
    ]
    if a.dry_run:
        print(json.dumps(commands, indent=2))
        return
    assert_free('npu', [int(x) for x in a.train_devices.split(',')])
    pre.mkdir(parents=True, exist_ok=False)
    cann = Path(os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/cann-9.1.0'))/'python/site-packages'
    env = os.environ.copy()
    env.update(DFLASH_PYTHON=a.python, DFLASH_TRAIN_DEVICES=a.train_devices,
               ASCEND_RT_VISIBLE_DEVICES=a.train_devices.split(',')[0],
               TORCHDYNAMO_DISABLE='1', TASK_QUEUE_ENABLE='1',
               PYTHONPATH=':'.join([str(BASE/'source/hs_connectors/src'), str(BASE/'source/src'), str(BASE/'scripts'), str(cann), env.get('PYTHONPATH','')]))
    for index, command in enumerate(commands):
        current = env.copy()
        if index == 0:
            current['AUDIT_OUT'] = str(pre/'CHECKPOINTING_PREFLIGHT.json')
        if index == 1:
            current['AUDIT_OUT'] = str(pre/'CPU_CHECKPOINT_PREFLIGHT.json')
            current['PREFLIGHT_ROOT'] = str(pre/'cpu_checkpoint_artifacts')
        with (pre/f'stage{index}.log').open('x') as log:
            subprocess.run(command, env=current, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Real-device preflights passed. The training supervisor will still check the full training topology.')


if __name__ == '__main__':
    main()
