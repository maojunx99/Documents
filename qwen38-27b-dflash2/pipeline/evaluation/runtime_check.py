"""Fail before serving if the selected inference Python lacks actual DFlash2 support."""
import argparse
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import write
from evaluation.core import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from vllm.model_executor.models import ModelRegistry
    from vllm.config import SpeculativeConfig
    architectures = ModelRegistry.get_supported_archs()
    if 'DFlash2DraftModel' not in architectures:
        raise RuntimeError('This inference environment has no registered DFlash2DraftModel. DFlash1 support is insufficient. '
                           'Set EVAL_PYTHON to an NPU runtime with DFlash2 model, selector and probabilistic-draft support; '
                           'the training environment is not automatically an inference environment.')
    fields = (set(getattr(SpeculativeConfig, 'model_fields', {})) |
              set(getattr(SpeculativeConfig, '__dataclass_fields__', {})) |
              set(getattr(SpeculativeConfig, '__annotations__', {})))
    if 'draft_sample_method' not in fields:
        raise RuntimeError('The selected runtime cannot explicitly select probabilistic draft sampling')
    if not torch.npu.is_available():
        raise RuntimeError('No NPU is visible to the inference Python')
    source = Path(importlib.util.find_spec('vllm').origin).parent
    implementations = [source / 'model_executor/models/qwen3_dflash2.py', source / 'v1/spec_decode/dflash.py']
    write(args.out, {'passed': True, 'python': sys.executable,
          'versions': {n: md.version(n) for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers']},
          'required_architecture': 'DFlash2DraftModel', 'draft_sample_method': True,
          'source_files': {str(p): sha(p) for p in implementations if p.is_file()},
          'scope': 'Capability check; actual requests and metrics must also pass.'})


if __name__ == '__main__':
    main()
