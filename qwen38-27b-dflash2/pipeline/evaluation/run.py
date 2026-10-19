"""Launch isolated NPU TP shards, evaluate T0/T1 and audit every real request."""
import json
import hashlib
import os
from pathlib import Path
import random
import signal
import socket
import time
import traceback

from common.managed import OwnedProcesses, assert_free, write
from evaluation.core import BENCHMARKS, aggregate, load_cases, serving_config, sha
from npu.recipe import device_groups

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(a, inspect_checkpoint):
    if a.eval_checkpoint:
        return a.eval_checkpoint.resolve()
    final = a.run / 'COMPLETION_VERIFICATION.json'
    if final.exists():
        record = json.loads(final.read_text())
        if record.get('passed'):
            return a.run / 'checkpoints' / str(record['completed_epochs'][-1])
    if inspect_checkpoint:
        raise ValueError('No completed native epoch; set EVAL_CHECKPOINT explicitly')
    return a.run / 'checkpoints' / '<last_completed_epoch>'


def plan(a, inspect_checkpoint=False):
    groups = device_groups(a.eval_groups)
    if min(a.eval_per_benchmark, a.eval_max_tokens, a.eval_max_model_len) < 1 or not 0 < a.eval_memory < 1:
        raise ValueError('Invalid evaluation budgets or memory fraction')
    if a.eval_max_tokens >= a.eval_max_model_len:
        raise ValueError('EVAL_CONTEXT must leave room for both input and output')
    if not 1 <= a.eval_base_port <= 65536-len(groups):
        raise ValueError('Invalid evaluation port range')
    if a.eval_cases is None:
        raise ValueError('Set EVAL_CASES to a five-benchmark JSON/JSONL case manifest')
    if len(groups) > len(BENCHMARKS)*a.eval_per_benchmark:
        raise ValueError('More inference groups than selected prompts')
    models = {'trained': str(checkpoint_path(a, inspect_checkpoint))}
    if a.eval_reference:
        models['reference'] = str(a.eval_reference.resolve())
    return {'models': models, 'cases': str(a.eval_cases.resolve()), 'groups': groups,
            'output': str(a.eval_output.resolve()) if a.eval_output else str(a.run / 'evaluations/<new_evaluation_run>'),
            'modes': ['t0', 't1'], 'python': a.eval_python or a.python,
            'expected_requests': len(models)*2*5*a.eval_per_benchmark,
            'per_benchmark': a.eval_per_benchmark, 'max_tokens': a.eval_max_tokens,
            'context': a.eval_max_model_len, 'presence_penalty': a.eval_presence_penalty,
            'C_per_instance': 1, 'BF16': True, 'spec_tokens': 7, 'enforce_eager': True,
            'thinking': True, 'reasoning_effort': a.eval_reasoning_effort,
            'manual_system_prompt': False, 'requires_dflash2_runtime_capability_check': True}


def export_checkpoint(source, destination):
    source = Path(source).resolve()
    config = serving_config(json.loads((source / 'config.json').read_text()))
    weights = sorted(source.glob('*.safetensors'))
    if not weights:
        raise ValueError('Checkpoint has no safetensors weights: ' + str(source))
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'config.json', config)
    manifest = {str(source / 'config.json'): sha(source / 'config.json')}
    for path in weights + sorted(source.glob('*.safetensors.index.json')):
        manifest[str(path)] = sha(path)
        # No multi-gigabyte duplicate and no in-place change to native checkpoints.
        (destination / path.name).symlink_to(path)
    write(destination / 'export_provenance.json', {'source': str(source), 'source_files': manifest,
          'serving_config_sha256': sha(destination / 'config.json'), 'weights_changed': False,
          'conversion': 'Native config flattened using vLLM DFlash2DraftModel field conventions; weight names/bytes unchanged.'})
    return manifest


def paired_comparison(rows, mode):
    by = {}
    for row in rows:
        if row['mode'] == mode:
            by.setdefault((row['benchmark'], row['case_id']), {})[row['model']] = row['counts']['tau']
    differences = [v['trained']-v['reference'] for v in by.values()
                   if v.get('trained') is not None and v.get('reference') is not None]
    if not differences:
        return None
    rng = random.Random(20261019)
    samples = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(10000))
    return {'paired_eligible_prompts': len(differences), 'trained_minus_reference': sum(differences)/len(differences),
            'prompt_bootstrap_95ci': [samples[249], samples[9749]], 'bootstrap_iterations': 10000,
            'scope': 'Prompt uncertainty at one generation seed; not independent training-seed replication.'}


def verify_rows(rows, cases, labels):
    expected = {(label, mode, c['benchmark'], c['case_id']): c for label in labels for mode in ['t0', 't1'] for c in cases}
    seen = set()
    for row in rows:
        key = (row['model'], row['mode'], row['benchmark'], row['case_id'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected evaluation request')
        case = expected[key]
        if row['prompt_hash'] != case['prompt_hash'] or row['seed'] != case['seed']:
            raise ValueError('Evaluation prompt/seed pairing changed')
        seen.add(key)
    if seen != set(expected):
        raise ValueError('Missing real evaluation requests')


def run_evaluation(a):
    protocol = plan(a, inspect_checkpoint=True)
    cases = load_cases(a.eval_cases, a.eval_per_benchmark)
    out = a.eval_output.resolve() if a.eval_output else a.run / 'evaluations' / f'eval_{time.time_ns()}_{os.getpid()}'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', protocol)
    write(out / 'selected_cases.json', cases)
    write(out / 'status.json', {'state': 'running', 'expected_requests': protocol['expected_requests']})
    cases_hash = sha(a.eval_cases)
    protected = {str(p): sha(p) for p in (ROOT / 'evaluation').glob('*.py')}
    for name in ['PIPELINE_CONFIG.json', 'COMPLETION_VERIFICATION.json', 'TRAINING_VERIFICATION.json']:
        path = a.run / name
        if path.is_file():
            protected[str(path)] = sha(path)
    write(out / 'training_provenance.json', {str(p): digest for p, digest in protected.items()
                                           if Path(p).parent == a.run})
    owned = OwnedProcesses(out)
    previous_signal = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        # Check capabilities before exporting or reserving any inference device.
        runtime = owned.start('runtime_check', [protocol['python'], str(ROOT / 'evaluation/runtime_check.py'),
                              '--out', str(out / 'runtime.json')], os.environ.copy(), cwd=ROOT)
        if runtime.wait(timeout=180) != 0:
            raise RuntimeError('The inference runtime failed its DFlash2 capability check; see ' + str(out / 'runtime_check.log'))
        assert_free('npu', [d for group in protocol['groups'] for d in group])
        for i in range(len(protocol['groups'])):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', a.eval_base_port+i))
        models, manifests = {}, {}
        training_receipt = a.run / 'stages/train/COMPLETED.json'
        expected_training = json.loads(training_receipt.read_text())['products'] if training_receipt.exists() else {}
        for label, checkpoint in protocol['models'].items():
            destination = out / 'exports' / label
            manifests.update(export_checkpoint(checkpoint, destination))
            if label == 'trained' and not a.eval_checkpoint:
                for name, digest in manifests.items():
                    if name in expected_training and expected_training[name] != digest:
                        raise ValueError('The selected final checkpoint differs from its training-stage fingerprint')
            models[label] = str(destination)
        write(out / 'model_fingerprints.json', manifests)
        workers = []
        count = len(protocol['groups'])
        for index, devices in enumerate(protocol['groups']):
            job = {'target': str(a.model), 'models': models, 'cases': cases[index::count],
                   'devices': devices, 'port': a.eval_base_port+index, 'context': a.eval_max_model_len,
                   'max_tokens': a.eval_max_tokens, 'memory': a.eval_memory,
                   'presence_penalty': a.eval_presence_penalty, 'reasoning_effort': a.eval_reasoning_effort,
                   'out': str(out / f'worker{index}')}
            path = out / f'job{index}.json'
            write(path, job)
            workers.append(owned.start(f'worker{index}', [protocol['python'], str(ROOT / 'evaluation/worker.py'),
                                       '--job', str(path)], os.environ.copy(), cwd=ROOT))
        deadline = time.monotonic()+a.stage_timeout
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in [None, 0] for p in workers):
                raise RuntimeError('An inference worker failed; retain its raw evidence and logs')
            if time.monotonic() > deadline:
                raise TimeoutError('Five-benchmark evaluation exceeded stage-timeout')
            time.sleep(1)
        if any(p.returncode for p in workers):
            raise RuntimeError('Inference worker exit failure')
        rows = []
        for index in range(count):
            worker = out / f'worker{index}'
            if not json.loads((worker / 'COMPLETED.json').read_text())['passed']:
                raise ValueError('Inference worker incomplete')
            rows.extend(json.loads(line) for line in (worker / 'per_request.jsonl').read_text().splitlines())
        verify_rows(rows, cases, models)
        # Recompute from the saved raw responses and metric snapshots, not just worker summaries.
        from evaluation.core import counters, parse_metrics, sampling
        audited = 0
        audited_rows = []
        for path in out.glob('worker*/*/t*/cases/*/result.json'):
            row = json.loads(path.read_text())
            before = parse_metrics((path.parent / 'metrics_before.txt').read_text())
            after = parse_metrics((path.parent / 'metrics_after.txt').read_text())
            if counters(before, after) != row['counts']:
                raise ValueError('Raw metrics re-audit differs')
            response = json.loads((path.parent / 'response.json').read_text())
            request = json.loads((path.parent / 'request.json').read_text())
            if response['usage'] != row['usage'] or request['messages'][0]['role'] != 'user' or len(request['messages']) != 1:
                raise ValueError('Response usage or message policy mismatch')
            if hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest() != row['prompt_hash']:
                raise ValueError('Saved request text differs from its paired original prompt')
            expected_sampling = sampling(row['mode'], a.eval_max_tokens, a.eval_presence_penalty)
            if any(request.get(k) != v for k, v in expected_sampling.items()) or request['seed'] != row['seed']:
                raise ValueError('Saved request sampling/seed differs from the protocol')
            if response['choices'][0]['finish_reason'] != row['finish_reason']:
                raise ValueError('Saved truncation/finish reason differs')
            audited_rows.append(row)
            audited += 1
        if audited != len(rows) or len(rows) != protocol['expected_requests']:
            raise ValueError('Incomplete saved raw request evidence')
        verify_rows(audited_rows, cases, models)
        key = lambda row: (row['model'], row['mode'], row['benchmark'], row['case_id'])
        if sorted(audited_rows, key=key) != sorted(rows, key=key):
            raise ValueError('Per-request index differs from independently audited evidence')
        if sha(a.eval_cases) != cases_hash or any(sha(path) != digest for path, digest in {**protected, **manifests}.items()):
            raise ValueError('Evaluation input, code or checkpoint changed')
        summary = {'output': str(out), 'expected_requests': protocol['expected_requests'], 'completed_requests': len(rows),
                   'raw_metrics_reaudited': audited, 'case_manifest_sha256': cases_hash, 'models': {},
                   'scope': 'Real inference acceptance/performance, not answer correctness or pass@1.'}
        for label in models:
            summary['models'][label] = {}
            for mode in ['t0', 't1']:
                subset = [r for r in rows if r['model'] == label and r['mode'] == mode]
                if not any(row['counts']['drafts'] > 0 for row in subset):
                    raise ValueError('No real speculative rounds observed for ' + label + '/' + mode)
                summary['models'][label][mode] = {'all': aggregate(subset), 'by_benchmark': {
                    b: aggregate([r for r in subset if r['benchmark'] == b]) for b in BENCHMARKS}}
        if 'reference' in models:
            summary['paired_comparisons'] = {mode: paired_comparison(rows, mode) for mode in ['t0', 't1']}
        write(out / 'summary.json', summary)
    except BaseException:
        write(out / 'failure.json', {'traceback': traceback.format_exc()})
        write(out / 'status.json', {'state': 'failed'})
        raise
    finally:
        try:
            owned.close()
        except BaseException:
            write(out / 'cleanup_failure.json', {'traceback': traceback.format_exc()})
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_signal)
    write(out / 'COMPLETED.json', {'passed': True, 'requests': len(rows), 'raw_metrics_reaudited': audited,
          'all_owned_workers_and_servers_released': True, 'summary_sha256': sha(out / 'summary.json')})
    write(out / 'status.json', {'state': 'completed', 'completed_requests': len(rows)})
    print(json.dumps(summary, indent=2))
    return summary
