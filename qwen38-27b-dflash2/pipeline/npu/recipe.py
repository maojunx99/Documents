"""CPU-only validation and command plans for the27B-compatible NPU workflow."""
import argparse
import copy
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def device_list(value):
    result = [int(x.strip()) for x in value.split(',') if x.strip()]
    if not result or any(x < 0 for x in result) or len(result) != len(set(result)):
        raise ValueError('Device lists must contain distinct nonnegative card IDs')
    return result


def device_groups(value):
    result = [device_list(x) for x in value.split(';')]
    flat = [i for group in result for i in group]
    if len(flat) != len(set(flat)):
        raise ValueError('A card occurs in more than one TP group')
    return result


def parser():
    p = argparse.ArgumentParser(description='One invocation: regenerate target answers -> prepare -> x*FSDPy -> validate/save/reload/audit.')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--input', type=Path, nargs='+', help='Raw conversations/messages JSONL; regenerate with this target')
    data.add_argument('--regenerated-dir', type=Path, help='Already regenerated tokenized shards; explicit reuse')
    p.add_argument('--python', default=__import__('sys').executable)
    p.add_argument('--replicas', type=int, default=2)
    p.add_argument('--shard-size', type=int, default=4)
    p.add_argument('--train-devices', default='8,9,10,11,12,13,14,15')
    p.add_argument('--teacher-groups', default='2,3', help='Independent TP groups separated by semicolons')
    p.add_argument('--regeneration-groups', default='8,9;10,11;12,13;14,15')
    p.add_argument('--generation-port', type=int, default=8500)
    p.add_argument('--teacher-port', type=int, default=8360)
    p.add_argument('--router-port', type=int, default=8350)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--presence-penalty', type=float, default=1.5)
    p.add_argument('--reasoning-effort', default='xhigh')
    p.add_argument('--generation-seed', type=int)
    p.add_argument('--generation-max-tokens', type=int, default=8192)
    p.add_argument('--generation-concurrency', type=int, default=34)
    p.add_argument('--generation-max-model-len', type=int, default=32768)
    p.add_argument('--generation-max-num-seqs', type=int, default=64)
    p.add_argument('--generation-memory-utilization', type=float, default=.85)
    p.add_argument('--draft-config', type=Path, default=BASE/'draft_config.json')
    p.add_argument('--target-layer-ids', type=int, nargs='+', default=[6,20,34,48,62])
    p.add_argument('--mask-token-id', type=int, default=248070)
    p.add_argument('--sample-max-length', type=int, default=4096)
    p.add_argument('--packed-tokens', type=int, default=4096)
    p.add_argument('--max-anchors', type=int, default=512)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--train-ratio', type=float, default=.99)
    p.add_argument('--lr', type=float, default=.0006)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--warmup-ratio', type=float, default=.04)
    p.add_argument('--selector-alpha', type=float, default=.1)
    p.add_argument('--ce-weight', type=float, default=.1)
    p.add_argument('--tv-weight', type=float, default=.9)
    p.add_argument('--gamma', type=float, default=4.)
    p.add_argument('--noise-std', type=float, default=.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--prepare-workers', type=int, default=8)
    p.add_argument('--teacher-max-num-seqs', type=int, default=4)
    p.add_argument('--teacher-memory-utilization', type=float, default=.80)
    p.add_argument('--hidden-cache', type=Path)
    p.add_argument('--stage-timeout', type=int, default=86400)
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(a):
    train = device_list(a.train_devices)
    teachers = device_groups(a.teacher_groups)
    regen = device_groups(a.regeneration_groups)
    if a.replicas < 1 or a.shard_size < 1 or len(train) != a.replicas*a.shard_size:
        raise ValueError('len(train-devices) must equal replicas*shard-size, with both positive')
    if set(train) & {i for group in teachers for i in group}:
        raise ValueError('Training and live teacher devices overlap')
    # Regeneration is a separate stage and may reuse all training/teacher cards.
    if a.temperature < 0 or not 0 < a.top_p <= 1 or (a.top_k != -1 and a.top_k < 1):
        raise ValueError('Invalid generation temperature/top-p/top-k')
    if not 0 < a.train_ratio < 1 or not 0 <= a.warmup_ratio < 1:
        raise ValueError('Require 0<train-ratio<1 and 0<=warmup-ratio<1')
    positive = ['sample_max_length','packed_tokens','max_anchors','epochs','prefetch_factor','prepare_workers',
                'generation_max_tokens','generation_concurrency','generation_max_num_seqs','teacher_max_num_seqs','stage_timeout']
    if any(getattr(a,n)<1 for n in positive) or a.workers < 1 or (a.max_steps is not None and a.max_steps < 1):
        raise ValueError('Lengths, budgets, workers, and prefetch values must be positive')
    if a.packed_tokens < a.sample_max_length:
        raise ValueError('packed-tokens must be >= sample-max-length')
    if not 0 < a.generation_memory_utilization < 1 or not 0 < a.teacher_memory_utilization < 1:
        raise ValueError('Memory utilization must be in (0,1)')
    if a.lr <= 0 or min(a.weight_decay,a.selector_alpha,a.ce_weight,a.tv_weight,a.noise_std)<0 or a.ce_weight+a.tv_weight<=0:
        raise ValueError('Invalid learning rate or loss weights')
    ports = [a.router_port]+list(range(a.teacher_port,a.teacher_port+len(teachers)))
    if len(ports)!=len(set(ports)) or any(not 1<=v<=65535 for v in ports+list(range(a.generation_port,a.generation_port+len(regen)))):
        raise ValueError('Invalid or overlapping live teacher/router ports')
    a.model=a.model.resolve();a.run=a.run.resolve();a.draft_config=a.draft_config.resolve()
    raw=json.loads((a.model/'config.json').read_text())
    target=raw.get('text_config',raw)
    draft=json.loads(a.draft_config.read_text())
    draft=draft.get('transformer_layer_config',draft)
    for name in ['hidden_size','vocab_size']:
        if target[name]!=draft[name]:
            raise ValueError(f'Target {name}={target[name]} differs from draft {draft[name]}; this example assumes the same27B structure. Update --draft-config deliberately.')
    if not 0 <= a.mask_token_id < target['vocab_size']:
        raise ValueError('mask-token-id is outside target vocabulary')
    if len(set(a.target_layer_ids))!=len(a.target_layer_ids) or any(i<0 or i>=target['num_hidden_layers'] for i in a.target_layer_ids):
        raise ValueError('Target feature layer IDs are invalid for this model')
    if a.sample_max_length>a.generation_max_model_len:
        raise ValueError('Generation context must accommodate training sample cap')
    return {'train':train,'teachers':teachers,'regeneration':regen,'target_text_config':target,'draft':copy.deepcopy(draft),
            'mesh':[list(range(i*a.shard_size,(i+1)*a.shard_size)) for i in range(a.replicas)]}


def command_plan(a, resolved):
    r=a.run;py=a.python;source=BASE/'source'
    plan={}
    if a.input:
        plan['generation_server']=[py,str(ROOT/'regeneration/serve.py'),'--backend','npu','--model',str(a.model),
          '--device-groups',a.regeneration_groups,'--run',str(r/'generation_servers'),'--base-port',str(a.generation_port),
          '--max-model-len',str(a.generation_max_model_len),'--max-num-seqs',str(a.generation_max_num_seqs),
          '--memory-utilization',str(a.generation_memory_utilization)]
        plan['regeneration']=[py,str(ROOT/'regeneration/regenerate.py'),'--model',str(a.model),'--input',*[str(p.resolve()) for p in a.input],
          '--endpoints',*[f'http://127.0.0.1:{a.generation_port+i}' for i in range(len(resolved['regeneration']))],
          '--run',str(r/'regenerated'),'--temperature',str(a.temperature),'--top-p',str(a.top_p),'--top-k',str(a.top_k),
          '--presence-penalty',str(a.presence_penalty),'--reasoning-effort',a.reasoning_effort,
          '--max-tokens',str(a.generation_max_tokens),'--concurrency',str(a.generation_concurrency)]
        if a.generation_seed is not None:plan['regeneration']+=['--generation-seed',str(a.generation_seed)]
        data=r/'regenerated/shards'
    else:data=a.regenerated_dir.resolve()
    plan['prepare']=[py,str(source/'scripts/prepare_data.py'),'--model',str(a.model),'--data',str(data/'*_out.jsonl'),
                     '--seq-length',str(a.sample_max_length),'--output',str(r/'prepared'),'--seed',str(a.seed),
                     '--num-preprocessing-workers',str(a.prepare_workers)]
    hidden=a.hidden_cache.resolve() if a.hidden_cache else r/'hidden_states'
    context=max(a.packed_tokens,a.sample_max_length)+8
    plan['teachers']=[]
    for i,group in enumerate(resolved['teachers']):
        plan['teachers'].append([py,str(source/'scripts/launch_vllm.py'),str(a.model),'--target-layer-ids',*map(str,a.target_layer_ids),
          '--hidden-states-path',str(hidden),'--','--tensor-parallel-size',str(len(group)),'--port',str(a.teacher_port+i),
          '--dtype','bfloat16','--enforce-eager','--no-enable-chunked-prefill','--gpu-memory-utilization',str(a.teacher_memory_utilization),
          '--max-num-seqs',str(a.teacher_max_num_seqs),'--max-num-batched-tokens',str(max(8192,context)),'--max-model-len',str(max(8192,context)),
          '--renderer-num-workers','1'])
    plan['router']=[py,str(ROOT/'gpu/teacher_router.py')]
    plan['parallel_preflight']=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(resolved['train'])),
      str(BASE/'scripts/preflight_parallel.py'),'--replicas',str(a.replicas),'--shard-size',str(a.shard_size),'--out',str(r/'preflight/parallel')]
    plan['trainer']=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(resolved['train'])),str(BASE/'scripts/run_train_parallel.py'),
      '--parallel-replicas',str(a.replicas),'--parallel-shard-size',str(a.shard_size),
      '--verifier-name-or-path',str(a.model),'--speculator-type','dflash2','--draft-config',str(r/'draft_config.json'),
      '--draft-attn-impl','sdpa','--data-path',str(r/'prepared'),'--hidden-states-backend','file','--hidden-states-path',str(hidden),
      '--vllm-endpoint',f'http://127.0.0.1:{a.router_port}/v1','--save-path',str(r/'checkpoints'),'--log-dir',str(r/'logs'),
      '--run-name',r.name,'--block-size','8','--max-anchors',str(a.max_anchors),'--target-layer-ids',*map(str,a.target_layer_ids),
      '--mask-token-id',str(a.mask_token_id),'--conv-kernel-size','2','--conv-group-size','16','--selector-rank','256','--selector-top-k','16',
      '--selector-loss-alpha',str(a.selector_alpha),'--loss-fn',json.dumps({'ce':a.ce_weight,'tv':a.tv_weight}),'--loss-implementation','fused',
      '--dflash-decay-gamma',str(a.gamma),'--per-position-loss-weight','fixed-exp-decay','--optimizer','adamw','--lr',str(a.lr),
      '--weight-decay',str(a.weight_decay),'--scheduler-type','cosine','--scheduler-warmup-ratio',str(a.warmup_ratio),'--epochs',str(a.epochs),
      '--total-seq-len',str(a.packed_tokens),'--train-data-ratio',str(a.train_ratio),'--seed',str(a.seed),
      '--num-workers',str(a.workers),'--prefetch-factor',str(a.prefetch_factor),'--noise-std',str(a.noise_std),
      '--hidden-states-dtype','bfloat16','--request-timeout','900','--max-retries','3','--on-missing','generate','--on-generate','delete',
      '--checkpoint-freq','1','--fsdp-shard','--log-freq','10','--no-resume-from-checkpoint']
    if a.max_steps is not None:plan['trainer']+=['--max-steps',str(a.max_steps)]
    return plan
