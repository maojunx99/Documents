"""CPU-only case loading, counter validation, checkpoint export and aggregation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

BENCHMARKS = ['gsm8k', 'math500', 'humaneval', 'mbpp', 'mt-bench']
METRIC = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path, per_benchmark):
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix.lower() == '.jsonl':
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data['cases']
    if not isinstance(data, list) or per_benchmark < 1:
        raise ValueError('Cases must be a JSON list, a {cases:[...]} object, or JSONL')
    groups = {b: [] for b in BENCHMARKS}
    seen = set()
    for index, source in enumerate(data):
        benchmark = str(source['benchmark']).lower().replace('_', '-').replace(' ', '')
        if benchmark == 'mtbench':
            benchmark = 'mt-bench'
        if benchmark not in groups:
            raise ValueError('Unknown benchmark: ' + benchmark)
        messages = source.get('messages')
        if messages is None:
            messages = [{'role': 'user', 'content': source['prompt']}]
        if not isinstance(messages, list) or len(messages) != 1 or messages[0].get('role') != 'user':
            raise ValueError('Evaluation accepts one original user message only; no system/few-shot additions')
        content = messages[0].get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Missing user prompt text')
        digest = hashlib.sha256(content.encode()).hexdigest()
        if source.get('prompt_hash', digest) != digest:
            raise ValueError('Case prompt hash mismatch')
        case_id = str(source.get('case_id', source.get('id', f'{benchmark}-{index}')))
        if (benchmark, case_id) in seen:
            raise ValueError('Duplicate benchmark/case ID')
        seen.add((benchmark, case_id))
        groups[benchmark].append({'benchmark': benchmark, 'case_id': case_id,
            'messages': [{'role': 'user', 'content': content}], 'prompt_hash': digest,
            'seed': int(source.get('seed', 2026101900+index))})
    if any(len(rows) < per_benchmark for rows in groups.values()):
        raise ValueError(f'Need at least {per_benchmark} cases in EACH of {BENCHMARKS}')
    return [row for name in BENCHMARKS for row in groups[name][:per_benchmark]]


def sampling(mode, max_tokens=4096, presence_penalty=0.):
    if mode not in ['t0', 't1']:
        raise ValueError('Expected t0 or t1')
    return {'temperature': 0. if mode == 't0' else 1.,
            'top_p': 1. if mode == 't0' else .95, 'top_k': -1 if mode == 't0' else 20,
            'presence_penalty': presence_penalty, 'max_tokens': max_tokens}


def parse_metrics(raw, spec_tokens=7):
    result = dict(drafts=0., draft_tokens=0., accepted=0., success=0., running=0., waiting=0.,
                  positions=[0.]*spec_tokens, present=[])
    mapping = {'vllm:spec_decode_num_drafts_total': 'drafts',
               'vllm:spec_decode_num_draft_tokens_total': 'draft_tokens',
               'vllm:spec_decode_num_accepted_tokens_total': 'accepted',
               'vllm:request_success_total': 'success', 'vllm:num_requests_running': 'running',
               'vllm:num_requests_waiting': 'waiting'}
    for line in raw.splitlines():
        match = METRIC.match(line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        if name not in mapping and name != 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid metric counter/gauge: ' + line)
        if name in mapping:
            key = mapping[name]
            result[key] += value
            if key not in result['present']:
                result['present'].append(key)
        else:
            position = re.search(r'(?:^|,)\s*position="(\d+)"', labels or '')
            if position is None or not 0 <= int(position[1]) < spec_tokens:
                raise ValueError('Invalid speculative position label')
            result['positions'][int(position[1])] += value
            if 'positions' not in result['present']:
                result['present'].append('positions')
    return result


def counters(before, after, spec_tokens=7):
    required = {'drafts', 'draft_tokens', 'accepted', 'success', 'running', 'waiting', 'positions'}
    if not required.issubset(after['present']):
        raise ValueError('Runtime does not expose the required speculative decoding metrics')
    if after['success']-before['success'] != 1 or after['running'] or after['waiting']:
        raise ValueError('An exclusive settled C1 request is required')
    result = {key: after[key]-before[key] for key in ['drafts', 'draft_tokens', 'accepted']}
    positions = [a-b for a, b in zip(after['positions'], before['positions'])]
    values = list(result.values()) + positions
    if any(x < 0 or abs(x-round(x)) > 1e-6 for x in values):
        raise ValueError('Counters reset or contain invalid increments')
    drafts = result['drafts']
    if not 0 <= result['accepted'] <= result['draft_tokens'] <= spec_tokens*drafts:
        raise ValueError('Draft/accepted counter bounds differ')
    if abs(sum(positions)-result['accepted']) > 1e-6:
        raise ValueError('Accepted-token and position counters disagree')
    if positions[0] > drafts or any(a < b for a, b in zip(positions, positions[1:])):
        raise ValueError('Survival counts must decrease by position')
    tau = 1+result['accepted']/drafts if drafts else None
    survival = [x/drafts for x in positions] if drafts else [None]*spec_tokens
    if tau is not None and abs(tau-1-sum(survival)) > 1e-9:
        raise ValueError('Acceptance-length identity failed')
    result.update(positions=positions, tau=tau, survival=survival,
                  conditional_acceptance=[n/d if d else None for n, d in zip(positions, [drafts]+positions[:-1])])
    return result


def serving_config(raw):
    if raw.get('dflash_config') and raw.get('architectures') in [
            ['DFlash2DraftModel'], ['DFlash2Qwen3ForCausalLM']]:
        result = copy.deepcopy(raw)
        result['architectures'] = ['DFlash2DraftModel']
        return result
    if raw.get('speculators_model_type') != 'dflash2':
        raise ValueError('Expected a native DFlash2 checkpoint or a DFlash2 serving export')
    if raw.get('block_size') != 8 or raw.get('sample_from_anchor', False):
        raise ValueError('This evaluation protocol requires block8/spec7 without anchor sampling')
    config = copy.deepcopy(raw['transformer_layer_config'])
    config['architectures'] = ['DFlash2DraftModel']
    config['draft_vocab_size'] = raw.get('draft_vocab_size')
    if raw.get('target_hidden_size') is not None:
        config['target_hidden_size'] = raw['target_hidden_size']
    # Native training indexes the hidden-state tuple (embedding=0); vLLM's
    # target_layer_ids are decoder-layer indexes and its runner adds one.
    native_layers = raw['aux_hidden_state_layer_ids']
    if not native_layers or any(i < 1 for i in native_layers):
        raise ValueError('Expected positive native auxiliary hidden-state indexes')
    config['eagle_aux_hidden_state_layer_ids'] = copy.deepcopy(native_layers)
    config['dflash_config'] = {'mask_token_id': raw['mask_token_id'],
        'target_layer_ids': [i-1 for i in native_layers], 'sample_from_anchor': False,
        'causal': not raw['sliding_window_non_causal']}
    for key in ['conv_kernel_size', 'conv_group_size', 'selector_rank', 'selector_top_k']:
        config['dflash_config'][key] = raw[key]
    return config


def aggregate(rows):
    if not rows:
        raise ValueError('No requests to aggregate')
    if any(r['elapsed_seconds'] <= 0 for r in rows):
        raise ValueError('Request timing must be positive')
    drafts = sum(r['counts']['drafts'] for r in rows)
    accepted = sum(r['counts']['accepted'] for r in rows)
    positions = [sum(r['counts']['positions'][i] for r in rows) for i in range(7)]
    valid = [r['counts']['tau'] for r in rows if r['counts']['tau'] is not None]
    tokens = sum(r['usage']['completion_tokens'] for r in rows)
    seconds = sum(r['elapsed_seconds'] for r in rows)
    return {'requests': len(rows), 'tau_eligible_requests': len(valid),
            'request_mean_tau': sum(valid)/len(valid) if valid else None,
            'global_draft_weighted_tau': 1+accepted/drafts if drafts else None,
            'survival': [n/drafts if drafts else None for n in positions],
            'conditional_acceptance': [n/d if d else None for n, d in zip(positions, [drafts]+positions[:-1])],
            'output_tokens': tokens, 'request_seconds_sum': seconds,
            'generation_tokens_per_second': tokens/seconds,
            'mean_per_request_tokens_per_second': sum(r['usage']['completion_tokens']/r['elapsed_seconds'] for r in rows)/len(rows),
            'length_truncated': sum(r['finish_reason'] == 'length' for r in rows),
            'metric_scope': 'C1 per-instance request timing, excludes server startup and metrics polling; not cluster aggregate throughput.'}
