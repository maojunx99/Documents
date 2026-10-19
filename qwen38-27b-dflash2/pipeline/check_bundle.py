"""CPU-only publication checks. No torch import, network, or accelerator work."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for relative, expected in manifest['files'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Published file changed: ' + relative)
    parsed = 0
    for p in ROOT.rglob('*.py'):
        if any(x in p.parts for x in ['runs', 'probes', 'preflight', '__pycache__']):
            continue
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        parsed += 1
    commands = [
        ['regeneration/serve.py', '--backend', 'npu', '--model', '/models/target', '--device-groups', '2,3', '--run', '/runs/servers', '--dry-run'],
        ['regeneration/regenerate.py', '--model', '/models/target', '--input', '/data/autoif_train_Qwen3-8B.jsonl',
         '--endpoints', 'http://127.0.0.1:8500', '--run', '/runs/answers', '--dry-run'],
        ['gpu/run_pipeline.py', '--model', '/models/target', '--data-dir', '/data/shards', '--run', '/runs/train', '--dry-run'],
        ['npu/scripts/prepare_preflights.py', '--dry-run'],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:]],
                                capture_output=True, text=True, check=True)
        json.loads(result.stdout)
    with tempfile.TemporaryDirectory() as folder:
        model=Path(folder)/'Qwen3.6-27B';model.mkdir()
        (model/'config.json').write_text(json.dumps({'text_config':{
            'hidden_size':5120,'vocab_size':248320,'num_hidden_layers':64}}))
        result=subprocess.run([sys.executable,str(ROOT/'npu/run_pipeline.py'),
            '--model',str(model),'--run',str(Path(folder)/'not_created'),
            '--input',str(Path(folder)/'prompts.jsonl'),'--dry-run'],capture_output=True,text=True,check=True)
        assert json.loads(result.stdout)['rank_mesh']==[[0,1,2,3],[4,5,6,7]]
        assert not (Path(folder)/'not_created').exists()
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True)
    sys.path.insert(0, str(ROOT / 'regeneration'))
    from verify_outputs import verify
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / 'source_00_out.jsonl'
        row = {'id': 'source_0_gen0', 'input_ids': [1, 2, 3], 'loss_mask': [0, 1, 1],
               'metadata': {'finish_reason': 'length', 'sampling_params': {'temperature': 1.0}}}
        p.write_text(json.dumps(row) + '\n', encoding='utf-8')
        assert verify(folder, expected_rows=1, expected_sampling={'temperature': 1.0})['rows'] == 1
        p.write_text((json.dumps(row) + '\n') * 2, encoding='utf-8')
        try:
            verify(folder)
        except ValueError as exc:
            assert 'Duplicate' in str(exc)
        else:
            raise AssertionError('Duplicate IDs must fail')
    print(json.dumps({'passed': True, 'hashes': len(manifest['files']), 'python_files_parsed': parsed,
                      'command_plans_checked': len(commands)+1, 'cpu_output_schema_and_duplicate_checks': True,
                      'npu_topology_model_switch_sampling_contract_tests': True,
                      'accelerator_jobs_started': 0}, indent=2))


if __name__ == '__main__':
    main()
