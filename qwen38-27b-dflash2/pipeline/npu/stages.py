"""Independent NPU stages with immutable configuration and verified artifact handoffs."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import socket
import time
import traceback
import uuid

from common.managed import OwnedProcesses, assert_free, identity, source_environment, write
from npu.run_pipeline import BASE, ROOT, health, protected_files, sha

ORDER = ['preflight', 'regenerate', 'prepare', 'train', 'evaluate']


@contextmanager
def locked(run, stage):
    token = uuid.uuid4().hex
    path = run / '.stage.lock'
    with path.open('x') as handle:
        json.dump({'token': token, 'identity': identity(os.getpid()), 'stage': stage}, handle)
    try:
        yield
    finally:
        if json.loads(path.read_text()).get('token') == token:
            path.unlink()


def serial_config(a):
    def encode(v):
        if isinstance(v, Path):
            return str(v.resolve())
        if isinstance(v, list):
            return [encode(x) for x in v]
        return v
    return {k: encode(v) for k, v in vars(a).items()}


def training_config(a):
    return {k: v for k, v in serial_config(a).items()
            if k not in ['stage', 'dry_run'] and not k.startswith('eval_')}


def target_manifest(model):
    return {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                     'sha256': sha(p) if p.stat().st_size < 1024*1024 else None}
            for p in model.iterdir() if p.is_file()}


def check_target(record):
    for name, item in record.items():
        p = Path(name)
        if not p.is_file() or p.stat().st_size != item['bytes'] or p.stat().st_mtime_ns != item['mtime_ns']:
            raise ValueError('Target file identity changed: ' + name)
        if item['sha256'] and sha(p) != item['sha256']:
            raise ValueError('Target configuration changed: ' + name)


def check_products(receipt):
    if not receipt.get('passed'):
        raise ValueError('Prerequisite stage has not passed')
    for name, expected in receipt['products'].items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError('Stage artifact changed: ' + name)


def previous_receipt(run, stage):
    if stage == 'preflight':
        return None
    name = ORDER[ORDER.index(stage)-1]
    path = run / 'stages' / name / 'COMPLETED.json'
    if not path.is_file():
        raise ValueError(f'Run the {name} stage successfully before {stage}')
    receipt = json.loads(path.read_text())
    check_products(receipt)
    return receipt


def stage_plan(a, resolved, commands):
    result = {'stage': a.stage, 'run': str(a.run), 'config': serial_config(a),
              'rank_mesh': resolved['mesh'], 'requires': None if a.stage == 'preflight' else ORDER[ORDER.index(a.stage)-1],
              'no_devices_or_files_created': True}
    keys = {'preflight': ['parallel_preflight'], 'regenerate': ['generation_server', 'regeneration'],
            'prepare': ['prepare'], 'train': ['teachers', 'router', 'trainer']}
    if a.stage == 'evaluate':
        from evaluation.run import plan
        result['evaluation'] = plan(a, inspect_checkpoint=False)
    else:
        result['commands'] = {k: commands[k] for k in keys[a.stage] if k in commands}
        if a.stage == 'regenerate' and a.regenerated_dir:
            result['reuse_verified_answers'] = str(a.regenerated_dir)
    return result


class Stage:
    def __init__(self, a, resolved, commands):
        self.a, self.resolved, self.commands = a, resolved, commands
        self.run = a.run
        self.folder = self.run / 'stages' / a.stage
        self.owners, self.live, self.products = [], [], []
        self.cache = None
        self.env = source_environment(BASE / 'source', {
            'TORCHDYNAMO_DISABLE': '1', 'TASK_QUEUE_ENABLE': '1', 'OMP_NUM_THREADS': '2',
            'HCCL_EXEC_TIMEOUT': '600', 'HCCL_CONNECT_TIMEOUT': '180',
            'TRAIN_AUDIT': str(self.run / 'audit'), 'NATIVE_SOURCE': str(BASE / 'source')})
        cann = Path(self.env.get('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.1.0')) / 'python/site-packages'
        self.env['PYTHONPATH'] = os.pathsep.join([str(BASE / 'scripts'), str(cann), self.env.get('PYTHONPATH', '')])
        self.train_env = dict(self.env, ASCEND_RT_VISIBLE_DEVICES=a.train_devices)

    def owner(self, name):
        folder = self.folder / name
        folder.mkdir(parents=True, exist_ok=False)
        owner = OwnedProcesses(folder)
        self.owners.append(owner)
        return owner

    def checked(self, name, argv, env=None):
        owner = self.owner(name)
        process = owner.start(name, argv, env or self.env, cwd=ROOT)
        deadline = time.monotonic() + self.a.stage_timeout
        while process.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError(name + ' exceeded stage-timeout')
            if any(p.poll() is not None for p in self.live):
                raise RuntimeError('A required service exited during ' + name)
            time.sleep(.5)
        if process.returncode:
            raise RuntimeError(name + ' failed; inspect ' + str(owner.run))

    def baseline(self):
        if protected_files(BASE / 'source') != json.loads((self.run / 'protected_source.json').read_text()):
            raise ValueError('Protected implementation changed between stages')
        check_target(json.loads((self.run / 'target_manifest.json').read_text()))
        for name, digest in json.loads((self.run / 'raw_input_fingerprints.json').read_text()).items():
            if sha(name) != digest:
                raise ValueError('Selected raw input changed: ' + name)

    def preflight(self):
        a, r = self.a, self.resolved
        (self.run / 'preflight').mkdir(exist_ok=False)
        cards = r['train'] + [d for g in r['teachers'] for d in g]
        assert_free('npu', cards)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
        vocab = tokenizer.get_vocab()
        if not vocab or max(vocab.values()) >= r['target_text_config']['vocab_size']:
            raise ValueError('Tokenizer exceeds the selected target vocabulary')
        mask = tokenizer.convert_ids_to_tokens(a.mask_token_id)
        if mask is None or tokenizer.convert_tokens_to_ids(mask) != a.mask_token_id:
            raise ValueError('Mask ID does not round trip')
        probe = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Compatibility check.'}],
                                              tokenize=True, add_generation_prompt=True)
        if hasattr(probe, 'keys'):
            probe = probe['input_ids']
        if not probe or any(not 0 <= i < r['target_text_config']['vocab_size'] for i in probe):
            raise ValueError('Invalid native chat template tokens')
        write(self.run / 'MODEL_COMPATIBILITY.json', {'passed': True, 'model': str(a.model),
              'mask_round_trip': True, 'native_template': True, 'tokenizer_class': type(tokenizer).__name__})
        if shutil.disk_usage(self.run).free < 25*2**30:
            raise RuntimeError('Need at least25GiB free for checkpoints and staging')
        for name, script in [('kernels', 'preflight_npu.py'), ('checkpointing', 'preflight_checkpointing.py'),
                             ('cpu_checkpoint', 'preflight_cpu_checkpoint.py')]:
            env = dict(self.env, ASCEND_RT_VISIBLE_DEVICES=str(r['train'][0]),
                       AUDIT_OUT=str(self.run / f'preflight/{name}.json'),
                       PREFLIGHT_ROOT=str(self.run / 'preflight/cpu_checkpoint_artifacts'))
            self.checked(name, [a.python, str(BASE / 'scripts' / script)], env)
            receipt = json.loads((self.run / f'preflight/{name}.json').read_text())
            if name == 'checkpointing':
                assert receipt['cpu']['passed'] and receipt['npu']['passed']
            else:
                assert receipt['passed']
            self.products.append(self.run / f'preflight/{name}.json')
        self.checked('parallel', self.commands['parallel_preflight'], self.train_env)
        for rank in range(a.replicas*a.shard_size):
            path = self.run / f'preflight/parallel/passed_rank{rank}.json'
            assert json.loads(path.read_text())['passed']
            self.products.append(path)

    def regenerate(self):
        a = self.a
        if a.input:
            assert_free('npu', [d for g in self.resolved['regeneration'] for d in g])
            owner = self.owner('generation_services')
            process = owner.start('servers', self.commands['generation_server'], self.env, cwd=ROOT)
            self.live.append(process)
            deadline = time.monotonic() + 1200*len(self.resolved['regeneration'])
            while not (self.run / 'generation_servers/endpoints.json').is_file():
                if process.poll() is not None:
                    raise RuntimeError('Generation server supervisor exited')
                if time.monotonic() > deadline:
                    raise TimeoutError('Generation readiness')
                time.sleep(.5)
            self.checked('regeneration', self.commands['regeneration'])
            assert json.loads((self.run / 'regenerated/COMPLETION.json').read_text())['passed']
            owner.close()
            nested = json.loads((self.run / 'generation_servers/cleanup.json').read_text())
            assert all(not item['remaining'] for item in nested)
            data = self.run / 'regenerated/shards'
        else:
            data = a.regenerated_dir.resolve()
            metadata = data.parent / 'REGENERATION_CONFIG.json'
            record = json.loads(metadata.read_text())
            if Path(record['model']).resolve() != a.model or not record.get('target_model_files'):
                raise ValueError('Reused answers require the same target and an identity manifest')
            check_target({str(a.model / name): item for name, item in record['target_model_files'].items()})
            write(self.run / 'answer_reuse.json', {'manifest': str(metadata), 'sha256': sha(metadata)})
        from regeneration.verify_outputs import verify
        result = verify(data)
        if min(int(result['rows']*a.train_ratio), result['rows']-int(result['rows']*a.train_ratio)) < a.replicas*a.shard_size:
            raise ValueError('Too few train/validation records for this topology')
        write(self.run / 'regenerated_data_verification.json', dict(result, shards=str(data)))
        self.products += [self.run / 'regenerated_data_verification.json', data.parent / 'REGENERATION_CONFIG.json']
        self.products += sorted(data.glob('*_out.jsonl'))

    def prepare(self):
        self.checked('prepare', self.commands['prepare'])
        from datasets import load_from_disk
        ds = load_from_disk(str(self.run / 'prepared'))
        lengths, supervised = [], 0
        for row in ds:
            ids, mask = row['input_ids'], row['loss_mask']
            assert 0 < len(ids) <= self.a.sample_max_length and len(ids) == len(mask) and any(mask)
            lengths.append(len(ids))
            supervised += sum(mask)
        train = int(len(ds)*self.a.train_ratio)
        if min(train, len(ds)-train) < self.a.replicas*self.a.shard_size:
            raise ValueError('Prepared train/validation split is too small')
        write(self.run / 'prepared_audit.json', {'rows': len(ds), 'input_tokens': sum(lengths),
              'supervised_tokens': int(supervised), 'max_length': max(lengths)})
        self.products = sorted(p for p in (self.run / 'prepared').rglob('*') if p.is_file())
        write(self.run / 'prepared_fingerprints.json', {str(p.relative_to(self.run / 'prepared')): sha(p) for p in self.products})
        self.products += [self.run / 'prepared_audit.json', self.run / 'prepared_fingerprints.json']

    def train(self):
        a, r = self.a, self.resolved
        assert_free('npu', r['train'] + [d for g in r['teachers'] for d in g])
        for port in [a.router_port] + list(range(a.teacher_port, a.teacher_port+len(r['teachers']))):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        cache = a.hidden_cache.resolve() if a.hidden_cache else self.run / 'hidden_states'
        if cache == self.run or self.run.is_relative_to(cache):
            raise ValueError('Hidden cache must not contain the run directory')
        cache.mkdir(parents=True, exist_ok=False)
        write(cache / 'OWNER.json', {'run': str(self.run), 'identity': identity(os.getpid())})
        self.cache = cache
        teachers = self.owner('teachers')
        for i, (group, command) in enumerate(zip(r['teachers'], self.commands['teachers'])):
            write(teachers.run / f'server{i}_provenance.json', {'argv': command, 'source_manifest': str(self.run / 'protected_source.json')})
            process = teachers.start(f'server{i}', command,
                       dict(self.env, ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, group))), cwd=BASE / 'source')
            self.live.append(process)
            health(a.teacher_port+i, process)
        env = dict(self.env, TEACHER_ROUTER_PORT=str(a.router_port),
                   TEACHER_BACKEND_PORTS=','.join(str(a.teacher_port+i) for i in range(len(r['teachers']))),
                   TEACHER_ROUTING_LOG=str(self.run / 'router_routing.jsonl'))
        router = teachers.start('router', self.commands['router'], env, cwd=ROOT)
        self.live.append(router)
        health(a.router_port, router, 60)
        self.checked('trainer', self.commands['trainer'], self.train_env)
        from npu.verify_run import verify_training
        result = verify_training(self.run, a, r)
        result.update(source_and_prepared_fingerprints_unchanged=True, target_manifest_unchanged=True)
        write(self.run / 'TRAINING_VERIFICATION.json', result)
        self.products = [self.run / 'TRAINING_VERIFICATION.json']
        self.products += sorted(p for p in (self.run / 'checkpoints').rglob('*') if p.is_file() and not p.is_symlink())

    def cleanup(self):
        errors = []
        for owner in reversed(self.owners):
            try:
                owner.close()
            except BaseException as exc:
                errors.append(repr(exc))
        if errors:
            write(self.folder / 'cleanup_failure.json', {'errors': errors})
            raise RuntimeError('Some owned processes did not cleanly exit; preserve the cache and inspect receipts')
        if self.cache:
            marker = json.loads((self.cache / 'OWNER.json').read_text())
            if marker['run'] != str(self.run):
                raise ValueError('Cache owner marker changed; no cleanup authorized')
            shutil.rmtree(self.cache)
            write(self.run / 'hidden_cache_cleanup.json', {'removed': str(self.cache), 'all_owned_consumers_stopped': True})


def execute(a, resolved, commands):
    if a.dry_run:
        print(json.dumps(stage_plan(a, resolved, commands), indent=2))
        return
    if os.name != 'posix':
        raise RuntimeError('Hardware stages require Linux; use --dry-run for CPU planning')
    run = a.run
    config_path = run / 'STAGED_CONFIG.json'
    if a.stage == 'preflight' and not run.exists():
        for path in a.input or []:
            if not path.is_file():
                raise ValueError('Missing raw input: ' + str(path))
        if a.regenerated_dir and not a.regenerated_dir.is_dir():
            raise ValueError('Missing regenerated directory')
        run.mkdir(parents=True, exist_ok=False)
        write(config_path, training_config(a))
        write(run / 'PIPELINE_CONFIG.json', serial_config(a))
        write(run / 'COMMAND_PLAN.json', commands)
        write(run / 'draft_config.json', resolved['draft'])
        write(run / 'protected_source.json', protected_files(BASE / 'source'))
        write(run / 'target_manifest.json', target_manifest(a.model))
        write(run / 'raw_input_fingerprints.json', {str(p.resolve()): sha(p) for p in a.input or []})
    if a.stage == 'evaluate':
        # Serving consumes target/checkpoint/cases, not the raw/prepared training data.
        # It can also follow an already-completed one-command training run.
        completion = json.loads((run / 'COMPLETION_VERIFICATION.json').read_text())
        if not completion['passed']:
            raise ValueError('The one-command training run has not passed verification')
        target = json.loads((run / 'target_manifest.json').read_text())
        check_target(target)
        saved = json.loads((run / 'PIPELINE_CONFIG.json').read_text())
        if Path(saved['model']).resolve() != a.model:
            raise ValueError('Evaluation target differs from the completed training target')
        if config_path.exists():
            if json.loads(config_path.read_text()) != training_config(a):
                raise ValueError('Keep the training configuration unchanged; edit only EVAL_* options')
            receipt = json.loads((run / 'stages/train/COMPLETED.json').read_text())
            if not receipt.get('passed'):
                raise ValueError('The training stage has not passed')
            for name in ['TRAINING_VERIFICATION.json', 'COMPLETION_VERIFICATION.json']:
                path = run / name
                if sha(path) != receipt['products'].get(str(path)):
                    raise ValueError('Training completion receipt changed')
        from evaluation.run import run_evaluation
        with locked(run, a.stage):
            result = run_evaluation(a)
            check_target(target)
            return result
    if not config_path.is_file() or json.loads(config_path.read_text()) != training_config(a):
        raise ValueError('Use the same fixed RUN_DIR and unchanged training configuration for every stage')
    stage = Stage(a, resolved, commands)
    stage.baseline()
    previous_receipt(run, a.stage)
    completed = stage.folder / 'COMPLETED.json'
    if completed.exists():
        check_products(json.loads(completed.read_text()))
        print(json.dumps({'stage': a.stage, 'already_completed': True, 'receipt': str(completed)}))
        return
    if stage.folder.exists():
        raise ValueError('This stage already has an unfinished/failed attempt; preserve it instead of overwriting or restarting')
    token = uuid.uuid4().hex
    lock = run / '.stage.lock'
    with lock.open('x') as handle:
        json.dump({'token': token, 'identity': identity(os.getpid()), 'stage': a.stage}, handle)
    try:
        stage.folder.mkdir(parents=True, exist_ok=False)
        write(stage.folder / 'launch_identity.json', identity(os.getpid()))
        write(stage.folder / 'status.json', {'state': 'running', 'stage': a.stage})
        previous_signal = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            getattr(stage, a.stage)()
            stage.baseline()
            previous_receipt(run, a.stage)
        finally:
            try:
                stage.cleanup()
            finally:
                signal.signal(signal.SIGTERM, previous_signal)
        if a.stage == 'train':
            result = json.loads((run / 'TRAINING_VERIFICATION.json').read_text())
            result.update(passed=True, owned_processes_released=True, hidden_cache_removed=True)
            write(run / 'COMPLETION_VERIFICATION.json', result)
            stage.products.append(run / 'COMPLETION_VERIFICATION.json')
        receipt = {'passed': True, 'stage': a.stage, 'utc_epoch': time.time(),
                   'products': {str(p): sha(p) for p in stage.products}, 'owned_processes_released': True}
        write(completed, receipt)
        write(stage.folder / 'status.json', {'state': 'completed', 'stage': a.stage})
        print(json.dumps(receipt, indent=2))
    except BaseException:
        write(stage.folder / 'failure.json', {'traceback': traceback.format_exc(), 'utc_epoch': time.time()})
        write(stage.folder / 'status.json', {'state': 'failed', 'stage': a.stage})
        raise
    finally:
        if json.loads(lock.read_text()).get('token') == token:
            lock.unlink()
