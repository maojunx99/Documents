"""Portable orchestration of the recorded 27B scratch recipe; validation is native."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import urllib.request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'common'))
sys.path.insert(0, str(BASE.parent))
from managed import OwnedProcesses, assert_free, source_environment, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True, help='Directory containing *_out.jsonl regeneration shards')
    p.add_argument('--run', type=Path, required=True, help='A new, nonexistent directory')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--teacher-devices', default='0,1,2,3')
    p.add_argument('--remote-teachers', type=Path, help='Prestarted native shared_file teachers: name/url/shared_root')
    p.add_argument('--train-devices', default='4,5,6,7')
    p.add_argument('--router-port', type=int, default=8250)
    p.add_argument('--teacher-base-port', type=int, default=8260)
    p.add_argument('--max-steps', type=int, help='Bounded smoke only; retains the original five-epoch schedule')
    p.add_argument('--selector-loss-type', choices=['hard_ce', 'full_teacher_tv'], default='hard_ce',
                   help='Selector objective; unary CE.1+TV.9 and alpha.1 stay unchanged')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    source = BASE / 'source'
    teachers = [int(x) for x in a.teacher_devices.split(',') if x.strip()]
    trainers = [int(x) for x in a.train_devices.split(',')]
    from teachers.protocol import load_peers
    peers = load_peers(a.remote_teachers)
    transfer_backend = 'shared_file' if peers else 'file'
    if (len(trainers) != 4 or len(set(teachers + trainers)) != len(teachers)+4
            or not (teachers or peers) or any(x < 0 for x in teachers+trainers)):
        p.error('Require four DDP trainers, disjoint local TP1 teachers and/or remote teachers')
    shards = sorted(a.data_dir.glob('*_out.jsonl'))
    if not shards and not a.dry_run:
        p.error('No *_out.jsonl data files found')
    prepare = [a.python, '-m', 'speculators', 'prepare-data', '--model', str(a.model)]
    for shard in (shards or [a.data_dir / 'SOURCE_00_out.jsonl']):
        prepare.extend(['--data', str(shard)])
    prepare += ['--output', str(a.run / 'prepared'), '--seq-length', '4096', '--seed', '42', '--num-preprocessing-workers', '32']
    train = [a.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4', '-m', 'speculators.train',
             '--verifier-name-or-path', str(a.model), '--data-path', str(a.run / 'prepared'),
             '--vllm-endpoint', f'http://127.0.0.1:{a.router_port}/v1', '--save-path', str(a.run / 'checkpoints'),
             '--log-dir', str(a.run / 'logs'), '--run-name', 'qwen38_27b_scratch', '--log-freq', '20',
             '--speculator-type', 'dflash2', '--block-size', '8', '--max-anchors', '1024',
             '--draft-config', str(BASE / 'draft_config.json'), '--mask-token-id', '248070',
             '--target-layer-ids', '6', '20', '34', '48', '62', '--loss-fn', '{"ce":0.1,"tv":0.9}',
             '--dflash-decay-gamma', '4', '--per-position-loss-weight', 'fixed-exp-decay', '--selector-loss-alpha', '.1',
             '--selector-loss-type', a.selector_loss_type,
             '--optimizer', 'adamw', '--lr', '.0006', '--weight-decay', '0', '--scheduler-type', 'cosine',
             '--scheduler-warmup-ratio', '.04', '--epochs', '5', '--total-seq-len', '8192', '--train-data-ratio', '.99',
             '--noise-std', '.05', '--seed', '42', '--hidden-states-dtype', 'bfloat16', '--hidden-states-backend', transfer_backend,
             '--hidden-states-path', str(a.run / 'hidden_states'), '--num-workers', '12', '--prefetch-factor', '4',
             '--on-missing', 'generate', '--on-generate', 'delete', '--request-timeout', '600', '--max-retries', '3',
             '--checkpoint-freq', '1', '--save-best', '--no-resume-from-checkpoint']
    if a.max_steps:
        train += ['--max-steps', str(a.max_steps)]
    teacher_commands = []
    for i, _ in enumerate(teachers):
        teacher_commands.append([a.python, str(source / 'scripts/launch_vllm.py'), str(a.model),
                                 '--target-layer-ids', '6', '20', '34', '48', '62', '--hidden-states-backend', transfer_backend,
                                 '--hidden-states-path', str(a.run / 'hidden_states'),
                                 '--provenance-dir', str(a.run / f'teacher{i}_provenance'), '--',
                                 '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '16384',
                                 '--mm-processor-cache-gb', '0', '--port', str(a.teacher_base_port+i)])
    if a.dry_run:
        print(json.dumps({'prepare': prepare, 'teachers': teacher_commands, 'remote_teachers': peers, 'train': train}, indent=2))
        return
    assert_free('gpu', teachers + trainers)
    a.run.mkdir(parents=True, exist_ok=False)
    (a.run / 'hidden_states').mkdir()
    env = source_environment(source)
    write(a.run / 'PIPELINE_CONFIG.json', {'model': str(a.model), 'raw_inputs': [str(x) for x in shards],
          'sample_cap': 4096, 'packed_per_rank': 8192, 'anchors': 1024, 'epochs': 5,
          'selector_loss_type': a.selector_loss_type, 'selector_loss_alpha': .1,
          'teacher_devices': teachers, 'training_devices': trainers, 'smoke_max_steps': a.max_steps,
          'remote_teachers': peers,
          'remote_config_sha256': hashlib.sha256(a.remote_teachers.read_bytes()).hexdigest() if a.remote_teachers else None,
          'initialization': 'Random draft; frozen target embedding/head; no existing draft checkpoint'})
    write(a.run / 'SOURCE_MANIFEST.json', {str(x.relative_to(source)): hashlib.sha256(x.read_bytes()).hexdigest()
          for x in source.rglob('*') if x.is_file() and '__pycache__' not in x.parts})
    owned = OwnedProcesses(a.run)
    remote_client = None
    teacher_workers = []
    result = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        owned.run_checked('prepare', prepare, env, source)
        for i, (device, cmd) in enumerate(zip(teachers, teacher_commands)):
            teacher_env = env.copy()
            teacher_env['CUDA_VISIBLE_DEVICES'] = str(device)
            worker = owned.start(f'teacher{i}', cmd, teacher_env, source)
            teacher_workers.append(worker)
            deadline = time.monotonic() + 1200
            while True:
                if worker.poll() is not None:
                    raise RuntimeError(f'Teacher {i} exited')
                try:
                    if urllib.request.urlopen(f'http://127.0.0.1:{a.teacher_base_port+i}/health', timeout=3).status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Teacher readiness')
                time.sleep(3)
        from teachers.client import connect, router_environment
        remote_backends = []
        if peers:
            remote_client, remote_backends = connect(peers, a.model, 'gpu', [6,20,34,48,62], a.run/'remote_teachers', 8193)
        router_env = dict(env, **router_environment(a.run, [a.teacher_base_port+i for i in range(len(teachers))], remote_backends))
        router_env.update(TEACHER_ROUTER_PORT=str(a.router_port),
                          TEACHER_ROUTING_LOG=str(a.run / 'router_routing.jsonl'))
        router = owned.start('router', [a.python, str(BASE / 'teacher_router.py')], router_env)
        for _ in range(30):
            if router.poll() is not None:
                raise RuntimeError('Teacher router exited')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{a.router_port}/health', timeout=3)
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Teacher router readiness')
        training_env = env.copy()
        training_env['CUDA_VISIBLE_DEVICES'] = a.train_devices
        trainer = owned.start('trainer', train, training_env, source)
        while trainer.poll() is None:
            if remote_client: remote_client.check()
            if router.poll() is not None: raise RuntimeError('Teacher router exited')
            if any(worker.poll() is not None for worker in teacher_workers): raise RuntimeError('Local teacher exited')
            time.sleep(.5)
        if trainer.returncode: raise RuntimeError('Trainer failed; inspect trainer.log')
        checkpoints = sorted(str(x) for x in (a.run / 'checkpoints').rglob('model.safetensors'))
        if not checkpoints:
            raise RuntimeError('Trainer exited without a checkpoint')
        result = {'native_train_exit': 0, 'checkpoints': checkpoints,
              'smoke': a.max_steps is not None, 'validation': 'Native trainer validates at each completed epoch',
              'checkpoint_limit': 'Native BF16 serialization and no RNG snapshot; not lossless FP32 resume'}
    except BaseException:
        write(a.run / 'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        try:
            owned.close()
        except BaseException:
            if remote_client: remote_client.close(consumers_stopped=False)
            raise
        if remote_client:
            try: remote_client.close(consumers_stopped=True)
            except BaseException:
                write(a.run/'failure.json', {'traceback':traceback.format_exc(), 'phase':'remote_cleanup'})
                raise
    if result is not None:
        write(a.run/'COMPLETION.json', dict(result, owned_processes_released=True, external_teachers_owned=False))


if __name__ == '__main__':
    main()
