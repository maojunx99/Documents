"""One exclusive TP service at a time; same prompt shard for T0/T1 and all checkpoints."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import OwnedProcesses, assert_free, identity, write
from evaluation.core import counters, parse_metrics, sampling


def http(port, path, body=None):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
              data=None if body is None else json.dumps(body).encode(),
              headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900 if body else 15) as response:
        return response.read()


def server_command(job, checkpoint, mode):
    speculative = {'method': 'dflash', 'model': checkpoint, 'num_speculative_tokens': 7,
                   'draft_sample_method': 'greedy' if mode == 't0' else 'probabilistic'}
    return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', job['target'],
            '--served-model-name', 'target', '--speculative-config', json.dumps(speculative),
            '--tensor-parallel-size', str(len(job['devices'])), '--dtype', 'bfloat16',
            '--enforce-eager', '--no-enable-prefix-caching', '--max-num-seqs', '1',
            '--max-model-len', str(job['context']), '--gpu-memory-utilization', str(job['memory']),
            '--port', str(job['port'])]


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'identity.json', identity(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(job['target'], local_files_only=True)
    rows = []
    for label, checkpoint in job['models'].items():
        for mode in ['t0', 't1']:
            folder = out / label / mode
            folder.mkdir(parents=True)
            owner = OwnedProcesses(folder)
            command = server_command(job, checkpoint, mode)
            parameters = sampling(mode, job['max_tokens'], job['presence_penalty'])
            write(folder / 'protocol.json', {'argv': command, 'sampling': parameters, 'C': 1,
                  'thinking': True, 'reasoning_effort': job['reasoning_effort'], 'manual_system_prompt': False})
            assert_free('npu', job['devices'])
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, job['devices'])),
                       TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1')
            try:
                server = owner.start('server', command, env, cwd=ROOT)
                deadline = time.monotonic()+1800
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Serving process exited; inspect server.log')
                    try:
                        http(job['port'], '/health')
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Inference readiness')
                        time.sleep(2)
                for index, case in enumerate(job['cases']):
                    evidence = folder / 'cases' / f'{index:04d}'
                    evidence.mkdir(parents=True)
                    kwargs = {'enable_thinking': True, 'reasoning_effort': job['reasoning_effort']}
                    prompt = tokenizer.apply_chat_template(case['messages'], tokenize=True,
                                                           add_generation_prompt=True, **kwargs)
                    if hasattr(prompt, 'keys'):
                        prompt = prompt['input_ids']
                    if len(prompt)+job['max_tokens'] > job['context']:
                        raise ValueError('Prompt+output exceeds EVAL_CONTEXT; inputs will not be silently truncated')
                    body = dict(model='target', messages=case['messages'], **parameters, seed=case['seed'],
                                stream=False, chat_template_kwargs=kwargs)
                    before_raw = http(job['port'], '/metrics').decode()
                    before = parse_metrics(before_raw)
                    if before['running'] or before['waiting']:
                        raise RuntimeError('Inference endpoint is not exclusive/idle')
                    write(evidence / 'request.json', body)
                    (evidence / 'metrics_before.txt').write_text(before_raw)
                    started = time.monotonic()
                    response = json.loads(http(job['port'], '/v1/chat/completions', body))
                    elapsed = time.monotonic()-started
                    write(evidence / 'response.json', response)
                    deadline = time.monotonic()+60
                    counter_error = None
                    while True:
                        after_raw = http(job['port'], '/metrics').decode()
                        after = parse_metrics(after_raw)
                        if after['success']-before['success'] == 1 and not after['running'] and not after['waiting']:
                            try:
                                counts = counters(before, after)
                            except ValueError as exc:
                                counter_error = exc
                            else:
                                break
                        if time.monotonic() > deadline:
                            (evidence / 'metrics_after.txt').write_text(after_raw)
                            raise TimeoutError('Per-request metrics did not settle or failed counter identities') from counter_error
                        time.sleep(.2)
                    (evidence / 'metrics_after.txt').write_text(after_raw)
                    if response['usage']['prompt_tokens'] != len(prompt):
                        raise ValueError('Runtime prompt length differs from the selected native tokenizer/template')
                    row = dict(case_id=case['case_id'], benchmark=case['benchmark'], prompt_hash=case['prompt_hash'],
                               seed=case['seed'], model=label, mode=mode, counts=counts, elapsed_seconds=elapsed,
                               usage=response['usage'], finish_reason=response['choices'][0]['finish_reason'])
                    write(evidence / 'result.json', row)
                    rows.append(row)
                    with (out / 'per_request.jsonl').open('a', encoding='utf-8') as target:
                        target.write(json.dumps(row)+'\n')
                    write(out / 'status.json', {'state': 'running', 'completed': len(rows),
                                               'expected': len(job['cases'])*2*len(job['models'])})
            finally:
                owner.close()
    expected = len(job['cases'])*2*len(job['models'])
    if len(rows) != expected:
        raise ValueError('Missing inference requests')
    write(out / 'COMPLETED.json', {'passed': True, 'requests': len(rows), 'owned_servers_released': True})
    write(out / 'status.json', {'state': 'completed', 'completed': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.job)
    except BaseException:
        job = json.loads(args.job.read_text())
        write(Path(job['out']) / 'failure.json', {'traceback': traceback.format_exc()})
        raise
