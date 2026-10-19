"""Record protected container packages and verify the selected NPU environment."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE_PACKAGES = ['torch', 'torch-npu', 'torchvision', 'torchaudio', 'vllm',
                 'vllm-ascend', 'triton-ascend', 'numpy']


def base_versions():
    versions = {name: md.version(name) for name in BASE_PACKAGES}
    expected = {'torch': '2.10.0', 'torch-npu': '2.10.0.post4',
                'vllm': '0.26.0', 'triton-ascend': '3.2.2'}
    for name, version in expected.items():
        if versions[name].split('+')[0] != version:
            raise ValueError(f'Expected {name}={version}, found {versions[name]}; check the image release')
    return versions


def record_base(out):
    versions = base_versions()
    out.mkdir(parents=True, exist_ok=True)
    previous = out / 'base-stack.json'
    if previous.exists() and json.loads(previous.read_text()) != versions:
        raise ValueError('The existing base-stack record differs; use a separate environment/record directory')
    (out / 'base-stack.txt').write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
    previous.write_text(json.dumps(versions, indent=2) + '\n')
    print(json.dumps(versions, indent=2))


def verify(out):
    versions = base_versions()
    if versions != json.loads((out / 'base-stack.json').read_text()):
        raise ValueError('A protected package changed during dependency installation')
    import torch
    import torch_npu  # noqa: F401
    import vllm  # noqa: F401
    import vllm_ascend  # noqa: F401
    import triton  # noqa: F401
    if not torch.npu.is_available():
        raise RuntimeError('The selected Python cannot access an NPU')
    # Separate interpreters prevent the second snapshot from reusing sys.modules.
    code = """import pathlib,sys,speculators,hs_connectors
root=pathlib.Path(sys.argv[1]).resolve()
paths=[pathlib.Path(m.__file__).resolve() for m in [speculators,hs_connectors]]
assert all(p.is_relative_to(root) for p in paths), paths
print(*(str(p) for p in paths),sep='\\n')
"""
    for snapshot in ['npu/source', 'regeneration/source']:
        source = ROOT / snapshot
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([
            str(source / 'src'), str(source / 'hs_connectors/src'), os.environ.get('PYTHONPATH', '')]))
        subprocess.run([sys.executable, '-c', code, str(source)], env=env, check=True)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out / 'python-freeze.txt').write_text(freeze)
    result = {'passed': True, 'python': sys.executable, 'base_packages': versions,
              'training_packages': {n: md.version(n) for n in ['transformers', 'datasets', 'pyarrow']},
              'npu_visible': True, 'source_snapshot_imports': True,
              'scope': 'Environment check only; run the bounded smoke before full training.'}
    (out / 'environment-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record-base', 'verify'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'record-base':
        record_base(args.out)
    else:
        verify(args.out)
