"""Start isolated generation servers on explicitly selected free devices."""
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from managed import OwnedProcesses, assert_free, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['gpu', 'npu'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--device-groups', required=True, help='Semicolon-separated TP groups, e.g. 0;1;2;3 or 2,3')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--base-port', type=int, default=8500)
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--max-num-seqs', type=int, default=64)
    ap.add_argument('--memory-utilization', type=float, default=.85)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    groups = [[int(x) for x in group.split(',')] for group in args.device_groups.split(';')]
    commands = []
    for i, devices in enumerate(groups):
        cmd = [args.python, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
               '--tensor-parallel-size', str(len(devices)), '--dtype', 'bfloat16',
               '--max-model-len', str(args.max_model_len), '--max-num-seqs', str(args.max_num_seqs),
               '--gpu-memory-utilization', str(args.memory_utilization), '--enforce-eager', '--port', str(args.base_port + i)]
        commands.append(cmd)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    assert_free(args.backend, [d for group in groups for d in group])
    args.run.mkdir(parents=True, exist_ok=False)
    owned = OwnedProcesses(args.run)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for i, (devices, cmd) in enumerate(zip(groups, commands)):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES' if args.backend == 'gpu' else 'ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            if args.backend == 'npu':
                env['TORCHDYNAMO_DISABLE'] = '1'
            p = owned.start(f'server{i}', cmd, env)
            deadline = time.monotonic() + 1200
            while True:
                if p.poll() is not None:
                    raise RuntimeError(f'Server {i} exited; inspect its log')
                try:
                    if urllib.request.urlopen(f'http://127.0.0.1:{args.base_port+i}/health', timeout=3).status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Generation server readiness timeout')
                time.sleep(3)
        endpoints = [f'http://127.0.0.1:{args.base_port+i}' for i in range(len(groups))]
        write(args.run / 'endpoints.json', endpoints)
        print(json.dumps({'ready': endpoints, 'stop': 'Ctrl+C in this foreground process'}, indent=2), flush=True)
        while all(p.poll() is None for p, _ in owned.processes):
            time.sleep(3)
        raise RuntimeError('A generation server exited')
    finally:
        owned.close()


if __name__ == '__main__':
    main()
