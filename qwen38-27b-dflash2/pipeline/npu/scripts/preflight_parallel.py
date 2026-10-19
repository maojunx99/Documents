"""Configurable x*FSDPy: collectives, global-batch gradient parity, BF16, save/reload."""
import argparse
import gc
import hashlib
import json
import os
import pathlib
import time
import traceback
from datetime import timedelta

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)

parser=argparse.ArgumentParser()
parser.add_argument('--replicas',type=int,required=True)
parser.add_argument('--shard-size',type=int,required=True)
parser.add_argument('--out',type=pathlib.Path,required=True)
options=parser.parse_args()
X,Y=options.replicas,options.shard_size
WORLD=X*Y
assert X>=1 and Y>=1
OUT=options.out
RANK = int(os.environ['RANK'])
LOCAL = int(os.environ['LOCAL_RANK'])


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def record(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}_rank{RANK}.json').write_text(json.dumps(value, indent=2))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 16)])
        self.head = nn.Linear(16, 8, bias=False)
        self.head.weight.requires_grad_(False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def inputs(rank):
    g = torch.Generator().manual_seed(20261019 + rank)
    return torch.randn(4, 32, generator=g), torch.randn(4, 8, generator=g)


try:
    torch.npu.set_device(LOCAL)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group('hccl', device_id=torch.device('npu', LOCAL), timeout=timedelta(seconds=180))
    assert dist.get_world_size() == WORLD
    mesh = init_device_mesh('npu',(X,Y) if X>1 else (Y,),mesh_dim_names=('replicate','shard') if X>1 else ('shard',))
    global_sum = torch.tensor(float(RANK + 1), device=f'npu:{LOCAL}')
    dist.all_reduce(global_sum)
    assert global_sum.item() == WORLD*(WORLD+1)//2
    checks = {}
    for dim in (['replicate','shard'] if X>1 else ['shard']):
        group = mesh.get_group(dim)
        members = dist.get_process_group_ranks(group)
        v = torch.tensor(float(RANK + 1), device=f'npu:{LOCAL}')
        dist.all_reduce(v, group=group)
        assert v.item() == sum(r + 1 for r in members)
        checks[dim] = members
    record('collectives', {'passed': True, 'mesh': mesh.mesh.tolist(), 'groups': checks})
    results = {}
    for dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(42)
        model = Toy()
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        original_sha = {n:digest(p) for n,p in original.items()}
        reference = Toy()
        reference.load_state_dict(original)
        policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        set_model_state_dict(model, {n:p.clone() for n,p in original.items()} if RANK == 0 else {}, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        assert all(digest(p.full_tensor()) == original_sha[n] for n, p in model.named_parameters())
        assert all(str(p.placements) == ('(Replicate(), Shard(dim=0))' if X>1 else '(Shard(dim=0),)') for p in model.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        reference_optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        x, y = inputs(RANK)
        x, y = x.to(f'npu:{LOCAL}'), y.to(f'npu:{LOCAL}')
        all_x, all_y = zip(*(inputs(r) for r in range(WORLD)))
        all_x, all_y = torch.cat(all_x), torch.cat(all_y)
        rows = []
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = (output.float() - y).square().mean()
            loss.backward()
            assert torch.isfinite(loss).item()
            differences = {}
            if dtype == torch.float32:
                reference_optimizer.zero_grad(set_to_none=True)
                reference_loss = (reference(all_x) - all_y).square().mean()
                reference_loss.backward()
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        full_grad = p.grad.full_tensor().cpu()
                        expected = dict(reference.named_parameters())[n].grad
                        torch.testing.assert_close(full_grad, expected, rtol=2e-4, atol=2e-5)
                        differences[n] = (full_grad - expected).abs().max().item()
                reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if dtype == torch.float32:
                torch.testing.assert_close(norm.full_tensor().cpu(), reference_norm, rtol=2e-4, atol=2e-5)
                reference_optimizer.step()
            optimizer.step()
            rows.append({'step': step + 1, 'loss': loss.item(), 'grad_max_abs_errors': differences})
        final = {n: digest(p.full_tensor()) for n, p in model.named_parameters()}
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, final)
        assert all(v == final for v in gathered)
        assert any(final[n] != original_sha[n] for n, p in model.named_parameters() if p.requires_grad)
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        optim = get_optimizer_state_dict(model, [optimizer], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        checkpoint = OUT / f'toy_{str(dtype).split(".")[-1]}.pt'
        if RANK == 0:
            torch.save({'model': full, 'optimizer': optim}, checkpoint)
        dist.barrier()
        with torch.no_grad():
            for p in model.parameters():
                p.to_local().add_(1)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False) if RANK == 0 else {'model': {}, 'optimizer': {}}
        set_model_state_dict(model, restored['model'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        set_optimizer_state_dict(model, [optimizer], restored['optimizer'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        assert all(digest(p.full_tensor()) == final[n] for n, p in model.named_parameters())
        results[str(dtype)] = {'steps': rows, 'save_reload_exact': True, 'all_ranks_same_full_model': True}
        del model, reference, optimizer, reference_optimizer, restored, optim, full
        gc.collect()
        torch.npu.empty_cache()
    record('passed', {'passed': True, 'world_size':WORLD, 'mesh': mesh.mesh.tolist(), 'groups': checks, 'results': results, 'utc_epoch': time.time()})
    dist.barrier()
    dist.destroy_process_group()
except BaseException:
    record('failure', {'traceback': traceback.format_exc(), 'utc_epoch': time.time()})
    raise
