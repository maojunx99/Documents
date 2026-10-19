"""Finite native regeneration jobs. Preserve the actual T1/p95/k20/xhigh recipe."""
from pathlib import Path
import argparse
import hashlib
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from managed import OwnedProcesses, source_environment, write

ALIASES = {'autoif_train_Qwen3-8B': 'autoif', 'evol_codealpaca_train_Qwen3-8B': 'evol_codealpaca',
           'lmsys_arena_train_Qwen3-8B': 'lmsys', 'metamathqa_train_Qwen3-8B': 'metamathqa',
           'orca_math_train_Qwen3-8B': 'orca_math', 'ultrachat_output': 'ultrachat',
           'ultrafeedback_train_sft_Qwen3-8B': 'ultrafeedback', 'ultrainteract_train_Qwen3-8B': 'ultrainteract'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', type=Path, nargs='+', required=True, help='ShareGPT conversations or OpenAI messages JSONL files')
    ap.add_argument('--endpoints', nargs='+', required=True, help='Server base URLs, e.g. http://127.0.0.1:8500')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--concurrency', type=int, default=34)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--presence-penalty', type=float, default=1.5)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=.95)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--reasoning-effort', default='xhigh')
    ap.add_argument('--generation-seed', type=int)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or not args.endpoints:
        ap.error('Positive concurrency/max-tokens and at least one endpoint are required')
    if args.temperature < 0 or not 0 < args.top_p <= 1 or (args.top_k != -1 and args.top_k < 1):
        ap.error('Require temperature>=0, 0<top-p<=1, and top-k=-1 or positive')
    sampling = {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k, 'presence_penalty': args.presence_penalty}
    if args.generation_seed is not None:
        sampling['seed'] = args.generation_seed
    source = Path(__file__).resolve().parent / 'source'
    def command(domain, i):
        return [args.python, '-m', 'speculators', 'regenerate-responses', '--model', args.model,
                '--endpoint', args.endpoints[i].rstrip('/') + '/v1/chat/completions',
                '--dataset', str(args.run / 'inputs' / f'{domain}_{i:02}.jsonl'),
                '--outfile', str(args.run / 'shards' / f'{domain}_{i:02}_out.jsonl'),
                '--concurrency', str(args.concurrency), '--max-tokens', str(args.max_tokens),
                '--sampling-params', json.dumps(sampling), '--reasoning-effort-cycle', args.reasoning_effort, '--resume']
    domains = [ALIASES.get(p.stem, p.stem) for p in args.input]
    if len(domains) != len(set(domains)):
        ap.error('Input files must have distinct source names')
    if args.dry_run:
        print(json.dumps([command(d, i) for d in domains for i in range(len(args.endpoints))], indent=2))
        return
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / 'inputs').mkdir()
    (args.run / 'shards').mkdir()
    manifest = {'model': args.model, 'sampling': sampling, 'reasoning_effort': args.reasoning_effort,
                'max_tokens': args.max_tokens, 'sources': {}, 'id_policy': 'source-prefixed IDs',
                'generation_seed': 'No explicit request seed; native regeneration metadata is retained.'}
    model_path = Path(args.model)
    if model_path.is_dir():
        manifest['target_model_files'] = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 1024*1024 else None}
            for p in model_path.iterdir() if p.is_file()}
    if args.generation_seed is not None:
        manifest['generation_seed'] = {'fixed_request_seed': args.generation_seed, 'scope': 'All requests use this explicit seed; not independent training-seed replication.'}
    for path, domain in zip(args.input, domains):
        handles = [(args.run / 'inputs' / f'{domain}_{i:02}.jsonl').open('x', encoding='utf-8') for i in range(len(args.endpoints))]
        count = 0
        digest = hashlib.sha256()
        try:
            with path.open('rb') as inp:
                for raw in inp:
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if not isinstance(row.get('conversations', row.get('messages')), list):
                        raise ValueError(f'{path}: expected conversations/messages; token-only data are not prompt inputs')
                    # Line number makes IDs unique even when different source rows reuse an ID.
                    row['id'] = f'{domain}_{count:09d}'
                    row['primary_id'] = row['id']
                    handles[count % len(handles)].write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
        finally:
            for handle in handles:
                handle.close()
        if count == 0:
            raise ValueError(f'Empty input: {path}')
        manifest['sources'][domain] = {'input': str(path.resolve()), 'rows': count, 'sha256': digest.hexdigest()}
    write(args.run / 'REGENERATION_CONFIG.json', manifest)
    env = source_environment(source)
    owned = OwnedProcesses(args.run)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    def lane(i):
        for domain in domains:
            if (args.run / 'inputs' / f'{domain}_{i:02}.jsonl').stat().st_size:
                owned.run_checked(f'client_{domain}_{i}', command(domain, i), env)
    try:
        pool = ThreadPoolExecutor(max_workers=len(args.endpoints))
        try:
            futures = [pool.submit(lane, i) for i in range(len(args.endpoints))]
            completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
            error = next((f.exception() for f in completed if f.exception() is not None), None)
            if error:
                owned.close()
                raise error
            for future in futures:
                future.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        import verify_outputs
        expected_ids = set()
        for path in (args.run/'inputs').glob('*.jsonl'):
            expected_ids.update(json.loads(line)['primary_id'] for line in path.read_text(encoding='utf-8').splitlines() if line)
        result = verify_outputs.verify(args.run / 'shards', expected_sampling=sampling, expected_primary_ids=expected_ids)
        write(args.run / 'COMPLETION.json', result)
        print(json.dumps(result, indent=2))
    finally:
        owned.close()


if __name__ == '__main__':
    main()
