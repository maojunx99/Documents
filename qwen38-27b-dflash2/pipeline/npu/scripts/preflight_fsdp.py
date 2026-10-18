"""Four real NPUs: FSDP2, AdamW, global clipping, full CPU save/reload."""
import hashlib
import json
import os
import pathlib
import time
import traceback

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)

ROOT = pathlib.Path(os.environ['PREFLIGHT_OUT'])
RANK = int(os.environ['RANK'])
LOCAL = int(os.environ['LOCAL_RANK'])


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64), nn.Linear(64, 64)])
        self.head = nn.Linear(64, 128, bias=False)
        self.head.weight.requires_grad_(False)

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x))
        return self.head(x)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


try:
    torch.npu.set_device(LOCAL)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group('hccl', device_id=LOCAL)
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = Toy()
    initial = {name: digest(p) for name, p in model.named_parameters()}
    state = model.state_dict() if RANK == 0 else {}
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False))
    del state
    assert all(digest(p.full_tensor()) == initial[name] for name, p in model.named_parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    torch.manual_seed(20261017 + RANK)
    x = torch.randn(8, 64, device=f'npu:{LOCAL}')
    labels = torch.randn(8, 128, device=f'npu:{LOCAL}')
    records = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('npu', dtype=torch.bfloat16):
            output = model(x)
            loss = (output.float() - labels).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert torch.isfinite(loss).item()
        records.append({'step': step + 1, 'loss': loss.item(), 'norm': float(norm.item()),
                        'allocated': torch.npu.memory_allocated(), 'peak': torch.npu.max_memory_allocated()})
    before = {name: digest(p.full_tensor()) for name, p in model.named_parameters()}
    assert any(before[name] != initial[name] for name, p in model.named_parameters() if p.requires_grad)
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    optim = get_optimizer_state_dict(model, [optimizer], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        torch.save({'model': full, 'optimizer': optim}, ROOT / 'toy_checkpoint.pt')
    dist.barrier()
    with torch.no_grad():
        for p in model.parameters():
            p.to_local().add_(1)
    restored = torch.load(ROOT / 'toy_checkpoint.pt', map_location='cpu', weights_only=False) if RANK == 0 else {'model': {}, 'optimizer': {}}
    set_model_state_dict(model, restored['model'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False))
    set_optimizer_state_dict(model, [optimizer], restored['optimizer'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    assert all(digest(p.full_tensor()) == before[name] for name, p in model.named_parameters())
    result = {'passed': True, 'rank': RANK, 'world': dist.get_world_size(), 'steps': records,
              'parameters': {name: {'global_shape': list(p.shape), 'local_shape': list(p.to_local().shape), 'dtype': str(p.dtype), 'device': str(p.device)} for name, p in model.named_parameters()},
              'full_model_reload_exact': True, 'full_optimizer_reload_executed': True, 'utc_epoch': time.time()}
    (ROOT / f'passed_rank{RANK}.json').write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    print(json.dumps(result), flush=True)
except BaseException:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f'failure_rank{RANK}.json').write_text(json.dumps({'traceback': traceback.format_exc(), 'utc_epoch': time.time()}, indent=2))
    raise
