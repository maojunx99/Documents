"""Named, strict replica checks with evidence written before a failure.

Parameter references must be captured after sharding, while they are the same
objects held by the optimizer. FSDP may expose different unsharded parameters
during a no-grad forward; do not rebuild this mapping after validation.
"""


def compare_replicas(payloads, replicas, shard_size):
    if len(payloads) != replicas * shard_size:
        raise ValueError('Replica audit world size does not match the mesh')
    differences = []
    for shard in range(shard_size):
        reference = payloads[shard]['tensors']
        for rank in range(shard + shard_size, len(payloads), shard_size):
            current = payloads[rank]['tensors']
            keys = sorted(set(reference) | set(current))
            changed = [key for key in keys if reference.get(key) != current.get(key)]
            if changed:
                differences.append({'ranks': [shard, rank], 'keys': changed})
    return differences


def make_payload(parameters, optimizers, local, digest):
    names = {parameter: name for name, parameter in parameters.items()}
    tensors = {'model/' + name: parameter for name, parameter in parameters.items()}
    scalars, order = {}, []
    for index, optimizer in enumerate(optimizers):
        ordered_names = []
        for parameter, state in optimizer.state.items():
            # The stable mapping is captured before the first optimizer step.
            name = names[parameter]
            ordered_names.append(name)
            for key, value in state.items():
                path = f'optimizer{index}/{name}/{key}'
                if hasattr(value, 'detach') and hasattr(value, 'dtype'):
                    tensors[path] = value
                else:
                    scalars[path] = {'kind': type(value).__name__, 'value': repr(value)}
        order.append(ordered_names)
    manifest = dict(scalars)
    for key, tensor in tensors.items():
        value = local(tensor)
        manifest[key] = {'kind': 'tensor', 'dtype': str(value.dtype),
                         'shape': list(value.shape), 'sha256': digest(value)}
    return {'tensors': manifest, 'optimizer_iteration_order': order}, tensors


def audit_replicas(parameters, optimizers, mesh, replicas, shard_size, save, phase):
    """Return exact equality; record numeric diagnostics without relaxing it."""
    import hashlib
    import time
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    def digest(value):
        value = value.detach().cpu().contiguous().reshape(-1)
        return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

    rank = dist.get_rank()
    payload, tensors = make_payload(parameters, optimizers, local, digest)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    differences = compare_replicas(gathered, replicas, shard_size)
    record = {'passed': not differences, 'phase': phase, 'utc_epoch': time.time(),
              'replicas': replicas, 'shard_size': shard_size,
              'all_rank_payloads': gathered, 'differences': differences,
              'criterion': 'Exact named local parameter and optimizer-state equality',
              'optimizer_order_is_diagnostic_only': True}
    # Preserve the full payload even if numerical collection later fails.
    save('replica_' + phase, record)
    dist.barrier()
    statistics = []
    if differences:
        source = rank % shard_size
        members = list(range(source, dist.get_world_size(), shard_size))
        group_differences = [item for item in differences if item['ranks'][0] == source]
        group_keys = sorted({key for item in group_differences for key in item['keys']})
        for key in group_keys:
            metadata = [gathered[member]['tensors'].get(key) for member in members]
            comparable = all(item and item['kind'] == 'tensor' for item in metadata)
            comparable = comparable and all((item['dtype'], item['shape']) ==
                                            (metadata[0]['dtype'], metadata[0]['shape'])
                                            for item in metadata)
            if not comparable:
                statistics.append({'key': key, 'numeric_comparison': 'incompatible metadata or scalar'})
                continue
            value = local(tensors[key]).detach()
            reference = value.clone().to(mesh.device_type)
            dist.broadcast(reference, src=source, group=mesh.get_group('replicate'))
            if rank != source and gathered[rank]['tensors'][key] != metadata[0]:
                x, y = value.to(mesh.device_type).float(), reference.float()
                delta = (x - y).abs()
                count = x.numel()
                statistics.append({'key': key, 'reference_rank': source, 'rank': rank,
                    'numel': count, 'unequal_elements': int((x != y).sum().item()),
                    'max_abs': delta.max().item() if count else 0.0,
                    'mean_abs': delta.mean().item() if count else 0.0,
                    'relative_l2': (torch.linalg.vector_norm(delta) /
                                    (torch.linalg.vector_norm(y) + 1e-30)).item() if count else 0.0,
                    'bf16_bitwise_equal': digest(x.bfloat16()) == digest(y.bfloat16()),
                    'first16_self': x.reshape(-1)[:16].cpu().tolist(),
                    'first16_reference': y.reshape(-1)[:16].cpu().tolist()})
            del reference
        record['numeric_differences_for_rank'] = statistics
        save('replica_' + phase, record)
    dist.barrier()
    return not differences
