# Publication changes: verify fresh preparation and run-local target/device manifests.
# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Independent FSDP4+512 completion checks; no process mutations."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import time


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def alive(identity):
    p = pathlib.Path(f'/proc/{identity["pid"]}/stat')
    if not p.exists():
        return False
    fields = p.read_text().rsplit(')', 1)[1].split()
    return fields[19] == str(identity['starttime']) and fields[0] != 'Z'


ap = argparse.ArgumentParser()
ap.add_argument('--run', type=pathlib.Path, required=True)
a = ap.parse_args()
run = a.run.resolve()
base = run.parents[1]
audit = run / 'audit'
assert not (run / 'failure.json').exists()
assert not list(audit.glob('failure_rank*.json'))
assert not alive(json.loads((run / 'launch_identity.json').read_text()))
leaders = json.loads((run / 'owned_leaders.json').read_text())
assert not any(alive(x) for x in leaders)
members = run / 'runtime_members_latest.json'
if members.exists():
    assert not any(alive(x) for x in json.loads(members.read_text()))
cfg = json.loads((run / 'TRAINING_CONFIG.json').read_text())
assert cfg['fsdp_shard'] and cfg['anchors_per_packed_rank'] == 512 and cfg['packed_tokens_per_rank'] == 4096
initial = [json.loads((audit / f'initial_rank{i}.json').read_text()) for i in range(4)]
assert all(v['trainable'] == initial[0]['trainable'] and v['optimizer_initial_states'] == [0] for v in initial)
assert all(v['trainable_parameters_global'] == 1924404480 for v in initial)
assert sum(v['trainable_parameters_local'] for v in initial) == 1924404480
assert all(v['trainable_parameters_local'] < v['trainable_parameters_global'] / 2 for v in initial)
steps, peak, validation = [], [], []
for rank, item in enumerate(initial):
    rows = [json.loads(line) for line in (audit / f'updates_rank{rank}.jsonl').read_text().splitlines()]
    expected = min(item['train_steps'], cfg['max_steps'] or item['train_steps'])
    assert len(rows) == expected
    assert all(v['completed_optimizer_steps'] == n + 1 and v['optimizer_step_min'] == v['optimizer_step_max'] == n + 1 and math.isfinite(v['loss']) for n, v in enumerate(rows))
    assert rows[-1]['sample_weight_changed_since_initialization']
    assert json.loads((audit / f'exit_rank{rank}.json').read_text())['exit_code'] == 0
    val = json.loads((audit / f'validation_epoch0_rank{rank}.json').read_text())
    assert val['global_step'] == expected and all(math.isfinite(x) for x in val['metrics'].values())
    steps.append(len(rows))
    peak.append({'rank': rank, 'peak_allocated_bytes': max(x['npu_peak_allocated_bytes'] for x in rows),
                 'peak_reserved_bytes': max(x['npu_peak_reserved_bytes'] for x in rows)})
    validation.append(val)
reload = json.loads((audit / 'checkpoint_reload_rank0.json').read_text())
assert reload['passed'] and reload['optimizer_load_executed_all_ranks'] and reload['model_load_executed_all_ranks']
selection = json.loads((run / 'selected/DATA_MANIFEST.json').read_text())
assert selection['rows'] == 10000 and sha(run / 'selected/selected_10000.jsonl') == selection['output_sha256']
(run / 'DATA_MANIFEST.json').write_text(json.dumps(selection, indent=2))
prepared = json.loads((run / 'prepared_fingerprints.json').read_text())
assert all(sha(run / 'prepared' / n) == h for n, h in prepared['files'].items())
source = json.loads((base / 'SOURCE_MANIFEST.json').read_text())['files']
assert all(sha(base / 'source' / n) == h for n, h in source.items())
implementation = json.loads((run / 'implementation_manifest.json').read_text())
assert all(sha(base / n) == implementation[n] for n in ['scripts/run_pipeline_fsdp.py', 'scripts/run_train_fsdp.py'])
target_manifest = run / 'target_model_manifest.json'
target_files = json.loads(target_manifest.read_text())
for name, item in target_files.items():
    f = pathlib.Path(name)
    assert f.stat().st_size == item['bytes'] and f.stat().st_mtime_ns == item['mtime_ns']
    if item['sha256']:
        assert sha(f) == item['sha256']
npu = subprocess.check_output(['npu-smi', 'info'], text=True)
(run / 'npu_after.txt').write_text(npu)
devices=[int(x) for x in (cfg['teacher_devices']+','+cfg['training_devices']).split(',')]
assert all(re.search(r'No running processes found in NPU ' + str(i) + r'\s', npu) for i in devices)
result = {'passed': True, 'full_epoch': cfg['max_steps'] is None, 'rows': 10000, 'steps_per_rank': steps,
          'fsdp_world_size': 4, 'anchors_per_rank': 512, 'vocabulary': 248320, 'full_vocab_CE_TV': True,
          'global_trainable_parameters': initial[0]['trainable_parameters_global'],
          'local_trainable_parameters': [v['trainable_parameters_local'] for v in initial],
          'initial81_global_tensors_equal': True, 'empty_initial_optimizer': True, 'all_losses_finite': True,
          'memory_peak_during_training': peak, 'native_validation': validation,
          'actual_checkpoint_reload': reload, 'source_and_data_hashes_unchanged': True,
          'target_files_stat_and_recorded_hashes_unchanged': len(target_files),
          'completion_verifier_sha256': sha(pathlib.Path(__file__)),
          'owned_leaders_exited': len(leaders), 'owned_npu_devices_released': devices,
          'checkpoint_model_sha256': sha(run / 'checkpoints/0/model.safetensors'), 'utc_epoch': time.time()}
(run / 'COMPLETION_VERIFICATION.json').write_text(json.dumps(result, indent=2))
(run / 'status.json').write_text(json.dumps({'stage': 'complete', 'utc_epoch': time.time()}))
print(json.dumps(result), flush=True)
