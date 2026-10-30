"""Read-only weight view for native and official DFlash2 warm starts."""
import copy
import hashlib
import json
from pathlib import Path


def describe(path, target):
    path, target = Path(path).resolve(), Path(target).resolve()
    raw = json.loads((path / 'config.json').read_text())
    verifier = json.loads((target / 'config.json').read_text())
    if raw.get('speculators_model_type') == 'dflash2':
        config = copy.deepcopy(raw)
    elif raw.get('architectures') == ['DFlash2DraftModel'] and 'dflash_config' in raw:
        d = raw['dflash_config']
        required = ['block_size', 'mask_token_id', 'conv_kernel_size', 'conv_group_size', 'selector_rank', 'selector_top_k', 'target_layer_ids']
        if any(k not in d for k in required):
            raise ValueError('Incomplete official DFlash2 configuration')
        excluded = {'architectures', 'auto_map', 'block_size', 'dflash_config', 'num_target_layers'}
        config = {
            'speculators_model_type': 'dflash2', 'architectures': ['DFlash2DraftModel'],
            'transformer_layer_config': {k: v for k, v in raw.items() if k not in excluded},
            'draft_vocab_size': raw['vocab_size'],
            # Official decoder layer indices refer to outputs; HF hidden_states[0]
            # is the embedding, so the training connector uses index + 1.
            'aux_hidden_state_layer_ids': [i + 1 for i in d['target_layer_ids']],
            'sliding_window_non_causal': True, 'sample_from_anchor': False,
            **{k: d[k] for k in required if k != 'target_layer_ids'},
        }
    else:
        raise ValueError('Expected native or official DFlash2 checkpoint, not a config-only decoder')
    config['speculators_config'] = {
        'algorithm': 'dflash2', 'proposal_methods': [{'proposal_type': 'greedy', 'speculative_tokens': config['block_size'] - 1}],
        'default_proposal_method': 'greedy',
        'verifier': {'name_or_path': str(target), 'architectures': verifier.get('architectures', [])},
    }
    weights = sorted(path.glob('model*.safetensors'))
    if not weights:
        raise ValueError('Warm start requires actual safetensors weights; config-only input is rejected')
    return {'path': str(path), 'config': config, 'weights': [str(p) for p in weights]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def materialize(a, resolved):
    spec = resolved.get('pretrained')
    if not spec:
        return
    out = a.run / 'pretrained_draft'
    source = Path(spec['path'])
    files = [source / 'config.json', *map(Path, spec['weights'])]
    index = source / 'model.safetensors.index.json'
    if index.exists():
        files.append(index)
    expected = {str(p): sha(p) for p in files}
    if out.exists():
        receipt = json.loads((out / 'WARM_START.json').read_text())
        if receipt['source_sha256'] != expected or json.loads((out / 'config.json').read_text()) != spec['config']:
            raise ValueError('Warm-start source or materialized configuration changed')
        return
    out.mkdir()
    (out / 'config.json').write_text(json.dumps(spec['config'], indent=2))
    for p in files:
        if p.name != 'config.json':
            (out / p.name).symlink_to(p)
    receipt = {'source': str(source), 'source_sha256': expected, 'configuration_only_adapter': True,
               'optimizer': 'new empty optimizer', 'scheduler': 'new CLI-configured schedule',
               'limitation': 'Weight-only fine tuning, not resumption of unpublished original optimizer/RNG.'}
    (out / 'WARM_START.json').write_text(json.dumps(receipt, indent=2))
