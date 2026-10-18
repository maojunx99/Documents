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
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or not args.endpoints:
        ap.error('Positive concurrency/max-tokens and at least one endpoint are required')
    sampling = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': args.presence_penalty}
    source = Path(__file__).resolve().parent / 'source'
    def command(domain, i):
        return [args.python, '-m', 'speculators', 'regenerate-responses', '--model', args.model,
                '--endpoint', args.endpoints[i].rstrip('/') + '/v1/chat/completions',
                '--dataset', str(args.run / 'inputs' / f'{domain}_{i:02}.jsonl'),
                '--outfile', str(args.run / 'shards' / f'{domain}_{i:02}_out.jsonl'),
                '--concurrency', str(args.concurrency), '--max-tokens', str(args.max_tokens),
                '--sampling-params', json.dumps(sampling), '--reasoning-effort-cycle', 'xhigh', '--resume']
    domains = [ALIASES.get(p.stem, p.stem) for p in args.input]
    if len(domains) != len(set(domains)):
        ap.error('Input files must have distinct source names')
    if args.dry_run:
        print(json.dumps([command(d, i) for d in domains for i in range(len(args.endpoints))], indent=2))
        return
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / 'inputs').mkdir()
    (args.run / 'shards').mkdir()
    manifest = {'model': args.model, 'sampling': sampling, 'reasoning_effort': 'xhigh',
                'max_tokens': args.max_tokens, 'sources': {}, 'id_policy': 'source-prefixed IDs',
                'generation_seed': 'No explicit request seed; native regeneration metadata is retained.'}
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
        result = verify_outputs.verify(args.run / 'shards', expected_rows=sum(v['rows'] for v in manifest['sources'].values()),
                                       expected_sampling=sampling)
        write(args.run / 'COMPLETION.json', result)
        print(json.dumps(result, indent=2))
    finally:
        owned.close()


if __name__ == '__main__':
    main()
