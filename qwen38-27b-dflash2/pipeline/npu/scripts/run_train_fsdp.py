"""Native FSDP4 training with full-vocabulary/512-anchor state and memory audits."""
import hashlib
import json
import math
import os
import pathlib
import runpy
import sys
import time
import traceback

import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from speculators.train.trainer import Trainer

AUDIT = pathlib.Path(os.environ['TRAIN_AUDIT'])
AUDIT.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ['RANK'])


def save(name, value):
    p = AUDIT / f'{name}_rank{RANK}.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


orig_model = Trainer.setup_model


def setup_model(self):
    assert self.config.fsdp_shard
    self.model.supports_gradient_checkpointing = True
    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    self._initial_cpu = {n: {'shape': list(p.shape), 'dtype': str(p.dtype), 'sha256': digest(p)}
                         for n, p in self.model.named_parameters() if p.requires_grad}
    save('before_shard', {'trainable': self._initial_cpu, 'random_initialization': True})
    orig_model(self)


orig_setup = Trainer.setup_optimizer


def setup(self):
    orig_setup(self)
    parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
    reconstructed = {n: {'shape': list(p.shape), 'dtype': str(p.dtype), 'sha256': digest(p.full_tensor())}
                     for n, p in parameters.items()}
    assert reconstructed == self._initial_cpu
    assert len(parameters) == 81 and all(isinstance(p, DTensor) for p in parameters.values())
    assert all(p.dtype == torch.float32 for p in parameters.values())
    assert all(len(opt.state) == 0 for opt in self.optimizers)
    self._audit_parameters = parameters
    self._sample_name = next(iter(parameters))
    self._sample = local(parameters[self._sample_name]).detach().flatten()[:4096].float().cpu().clone()
    self._audit_started = time.time()
    self._audit_loss = None
    state = {'rank': RANK, 'world_size': dist.get_world_size(), 'train_steps': len(self.train_loader),
             'val_steps': len(self.val_loader), 'trainable': reconstructed,
             'trainable_parameters_global': sum(p.numel() for p in parameters.values()),
             'trainable_parameters_local': sum(local(p).numel() for p in parameters.values()),
             'optimizer_initial_states': [len(opt.state) for opt in self.optimizers],
             'shards': {n: {'global': list(p.shape), 'local': list(local(p).shape), 'placements': str(p.placements), 'dtype': str(p.dtype)} for n, p in self.model.named_parameters()},
             'device_type': self.device_type, 'current_epoch': self.current_epoch,
             'npu_allocated_bytes': torch.npu.memory_allocated(), 'npu_reserved_bytes': torch.npu.memory_reserved()}
    assert state['device_type'] == 'npu' and state['current_epoch'] == 0
    save('initial', state)
    torch.npu.reset_peak_memory_stats()

    def output_hook(module, args, output):
        self._audit_loss = float(output[1].detach().float().item())
        assert math.isfinite(self._audit_loss), 'Nonfinite loss'

    def input_hook(module, args, kwargs):
        error = kwargs.get('error_records', 0)
        if isinstance(error, torch.Tensor):
            error = error.item()
        assert not error, f'Dropped input records: {error}'

    self.model.register_forward_hook(output_hook)
    self.model.register_forward_pre_hook(input_hook, with_kwargs=True)


orig_step = Trainer._optimizers_step


def step(self):
    n = self.global_step + 1
    gradients = [torch.isfinite(local(p.grad)).all() for p in self._audit_parameters.values() if p.grad is not None]
    assert gradients and torch.stack(gradients).all().item(), 'Nonfinite local gradient'
    lr = self.optimizers[0].param_groups[0]['lr']
    orig_step(self)
    counters = [float(s['step'].item()) for opt in self.optimizers for s in opt.state.values() if 'step' in s]
    assert counters and min(counters) == max(counters) == n
    changed = bool((local(self._audit_parameters[self._sample_name]).detach().flatten()[:4096].float().cpu() != self._sample).any())
    buffers = {}
    for opt in self.optimizers:
        for s in opt.state.values():
            for v in s.values():
                if isinstance(v, torch.Tensor):
                    v = local(v)
                    k = str(v.dtype) + '/' + str(v.device.type)
                    buffers[k] = buffers.get(k, 0) + v.numel() * v.element_size()
    row = {'rank': RANK, 'completed_optimizer_steps': n, 'loss': self._audit_loss, 'lr_used': lr,
           'optimizer_states': len(counters), 'optimizer_step_min': min(counters), 'optimizer_step_max': max(counters),
           'sample_weight_changed_since_initialization': changed, 'optimizer_local_bytes_by_dtype': buffers,
           'npu_allocated_bytes': torch.npu.memory_allocated(), 'npu_reserved_bytes': torch.npu.memory_reserved(),
           'npu_peak_allocated_bytes': torch.npu.max_memory_allocated(), 'npu_peak_reserved_bytes': torch.npu.max_memory_reserved(),
           'elapsed_seconds': time.time() - self._audit_started, 'utc_epoch': time.time()}
    with (AUDIT / f'updates_rank{RANK}.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    save('progress', row)


orig_val = Trainer.val_epoch


def val(self, epoch):
    metrics = orig_val(self, epoch)
    assert metrics and all(math.isfinite(v) for v in metrics.values())
    save(f'validation_epoch{epoch}', {'global_step': self.global_step, 'metrics': metrics})
    return metrics


orig_run = Trainer.run_training


def audited_run(self):
    orig_run(self)
    dist.barrier()
    # Collect full state on CPU before changing the live model; no NPU-wide copy.
    full = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        from safetensors.torch import load_file
        saved = pathlib.Path(self.config.save_path) / '0'
        serialized = load_file(str(saved / 'model.safetensors'), device='cpu')
        assert all(digest(full[n].to(torch.bfloat16)) == digest(serialized[n]) for n in self._audit_parameters)
        expected = {n: digest(serialized[n].float()) for n in self._audit_parameters}
        optimizer = torch.load(saved / 'optimizer_state_dict.pt', map_location='cpu', weights_only=False)
        assert len(optimizer['state']) == 81
        saved_steps = sorted(set(float(s['step'].item()) for s in optimizer['state'].values()))
        del optimizer, serialized
    else:
        expected, saved_steps = None, None
    del full
    dist.barrier()
    self.checkpointer.previous_epoch = 0
    self.checkpointer.load_model_state_dict(self.model)
    self.checkpointer.load_optimizer_state_dict(self.model, self.optimizers)
    hashes = {n: digest(p.full_tensor()) for n, p in self._audit_parameters.items()}
    save('reloaded_hashes', hashes)
    dist.barrier()
    if RANK == 0:
        gathered = [json.loads((AUDIT / f'reloaded_hashes_rank{i}.json').read_text()) for i in range(dist.get_world_size())]
        assert all(h == expected for h in gathered)
        save('checkpoint_reload', {'passed': True, 'model_load_executed_all_ranks': True,
             'optimizer_load_executed_all_ranks': True, 'trainable_tensors': 81,
             'all_ranks_match_serialized_bf16_expanded_fp32': True, 'saved_optimizer_step_values': saved_steps,
             'live_global_step_before_reload': self.global_step,
             'limitation': 'Native BF16 checkpoint rounds model/moments/step and contains no RNG snapshot; not lossless resume.'})
    dist.barrier()
    save('final', {'completed_optimizer_steps': self.global_step, 'expected_train_steps': len(self.train_loader),
                   'max_steps': self.config.max_steps, 'passed': True})


Trainer.setup_model = setup_model
Trainer.setup_optimizer = setup
Trainer._optimizers_step = step
Trainer.val_epoch = val
Trainer.run_training = audited_run
try:
    script = pathlib.Path(os.environ['NATIVE_SOURCE']) / 'scripts/train.py'
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name='__main__')
    save('exit', {'exit_code': 0, 'utc_epoch': time.time()})
except BaseException:
    save('failure', {'traceback': traceback.format_exc(), 'utc_epoch': time.time()})
    raise
