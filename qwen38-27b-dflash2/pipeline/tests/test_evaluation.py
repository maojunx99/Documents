"""Counter semantics, prompt pairing, native serving export and simulated real-request orchestration."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from evaluation import core, run, worker
from npu.recipe import parser, validate


def raw_metrics(drafts, accepted, positions, success, draft_tokens=None):
    values = {'spec_decode_num_drafts_total': drafts,
              'spec_decode_num_draft_tokens_total': 7*drafts if draft_tokens is None else draft_tokens,
              'spec_decode_num_accepted_tokens_total': accepted, 'request_success_total': success,
              'num_requests_running': 0, 'num_requests_waiting': 0}
    return '\n'.join(f'vllm:{name}{{model_name="target"}} {value}' for name, value in values.items()) + '\n' + '\n'.join(
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="target",position="{i}"}} {n}'
        for i, n in enumerate(positions))


def native_config():
    return {'speculators_model_type': 'dflash2', 'architectures': ['DFlash2DraftModel'],
            'transformer_layer_config': {'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 5},
            'block_size': 8, 'sample_from_anchor': False, 'aux_hidden_state_layer_ids': [6, 20, 34, 48, 62],
            'mask_token_id': 248070, 'conv_kernel_size': 2, 'conv_group_size': 16,
            'selector_rank': 256, 'selector_top_k': 16, 'sliding_window_non_causal': True}


class EvaluationContracts(unittest.TestCase):
    def test_request_mean_is_not_draft_weighted_and_conditional_need_not_decrease(self):
        zero = core.parse_metrics(raw_metrics(0, 0, [0]*7, 0))
        a = core.counters(zero, core.parse_metrics(raw_metrics(10, 30, [8, 6, 4, 4, 3, 3, 2], 1, 65)))
        b = core.counters(zero, core.parse_metrics(raw_metrics(100, 200, [60, 40, 30, 20, 20, 20, 10], 1)))
        self.assertEqual((a['tau'], b['tau']), (4., 3.))
        self.assertLess(a['conditional_acceptance'][0], a['conditional_acceptance'][3])
        rows = [{'counts': v, 'usage': {'completion_tokens': 50}, 'elapsed_seconds': 1., 'finish_reason': 'length'} for v in [a, b]]
        summary = core.aggregate(rows)
        self.assertEqual(summary['request_mean_tau'], 3.5)
        self.assertAlmostEqual(summary['global_draft_weighted_tau'], 1+230/110)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            core.counters(zero, core.parse_metrics(raw_metrics(10, 29, [8, 6, 4, 4, 3, 3, 2], 1)))

    def test_native_hidden_state_indexes_and_noncausal_attention_survive_export(self):
        raw = native_config()
        converted = core.serving_config(raw)
        self.assertEqual(converted['dflash_config']['target_layer_ids'], [5, 19, 33, 47, 61])
        self.assertFalse(converted['dflash_config']['causal'])
        self.assertEqual(raw['architectures'], ['DFlash2DraftModel'])
        self.assertEqual(converted['architectures'], ['DFlash2DraftModel'])

    def test_cases_preserve_user_text_seed_and_reject_system_or_missing_benchmarks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cases.json'
            cases = [{'benchmark': b, 'case_id': b, 'seed': 7,
                      'messages': [{'role': 'user', 'content': 'Original question '+b}]} for b in core.BENCHMARKS]
            path.write_text(json.dumps({'cases': cases}))
            selected = core.load_cases(path, 1)
            self.assertEqual([r['messages'] for r in selected], [r['messages'] for r in cases])
            self.assertTrue(all(r['seed'] == 7 for r in selected))
            cases[0]['messages'].insert(0, {'role': 'system', 'content': 'Do not inject this'})
            path.write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, 'no system'):
                core.load_cases(path, 1)

    def test_complete_serving_orchestration_with_cpu_http_and_process_standins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'target';model.mkdir()
            (model / 'config.json').write_text(json.dumps({'text_config': {'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 64}}))
            checkpoint = root / 'checkpoint';checkpoint.mkdir()
            (checkpoint / 'config.json').write_text(json.dumps(native_config()))
            (checkpoint / 'model.safetensors').write_bytes(b'CPU test bytes, never load into a model')
            cases = root / 'cases.json'
            cases.write_text(json.dumps([{'benchmark': b, 'id': b, 'prompt': 'Original '+b} for b in core.BENCHMARKS]))
            a = parser().parse_args(['--model', str(model), '--run', str(root / 'run'), '--input', str(root / 'unused'),
                '--eval-cases', str(cases), '--eval-checkpoint', str(checkpoint), '--eval-per-benchmark', '1',
                '--eval-groups', '0,1;2,3', '--eval-output', str(root / 'evaluation')])
            validate(a)
            closed = []
            counters_by_port = {}

            class Process:
                returncode = 0
                def poll(self): return 0
                def wait(self, timeout=None): return 0

            class Server:
                def poll(self): return None

            class ServerOwner:
                def __init__(self, folder): self.folder = folder
                def start(self, name, command, env, cwd=None):
                    spec = json.loads(command[command.index('--speculative-config')+1])
                    self.spec = spec['draft_sample_method']
                    return Server()
                def close(self):
                    closed.append(self.spec)
                    worker.write(self.folder / 'cleanup.json', [])

            class ParentOwner:
                def __init__(self, folder): self.folder = folder
                def start(self, name, command, env, cwd=None):
                    if name == 'runtime_check':
                        run.write(self.folder / 'runtime.json', {'passed': True, 'test_only': True})
                    else:
                        worker.main(command[-1])
                    return Process()
                def close(self): run.write(self.folder / 'cleanup.json', [])

            def http(port, path, body=None):
                number = counters_by_port.setdefault(port, 0)
                if path == '/health': return b''
                if path == '/metrics':
                    return raw_metrics(number, number*2, [number, number, 0, 0, 0, 0, 0], number).encode()
                self.assertEqual(len(body['messages']), 1)
                self.assertEqual(body['messages'][0]['role'], 'user')
                counters_by_port[port] += 1
                return json.dumps({'usage': {'prompt_tokens': 2, 'completion_tokens': 3},
                                   'choices': [{'finish_reason': 'stop', 'message': {'content': 'CPU fake response'}}]}).encode()

            tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kw: [1, 2])
            clock = itertools.count(100., .01)
            transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kw: tokenizer))
            # Hard links stand in for symlinks on Windows; source bytes stay unchanged.
            def link(destination, source):
                import os
                os.link(source, destination)
            with patch.object(run, 'OwnedProcesses', ParentOwner), patch.object(worker, 'OwnedProcesses', ServerOwner), \
                 patch.object(run, 'assert_free'), patch.object(worker, 'assert_free'), patch.object(worker, 'http', http), \
                 patch.object(run.socket, 'socket'), \
                 patch.object(worker.time, 'monotonic', side_effect=lambda: next(clock)), \
                 patch.dict('sys.modules', {'transformers': transformers}), patch.object(Path, 'symlink_to', link), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = run.run_evaluation(a)
            self.assertEqual(summary['completed_requests'], 10)
            self.assertEqual(summary['raw_metrics_reaudited'], 10)
            self.assertEqual(sorted(closed), ['greedy', 'greedy', 'probabilistic', 'probabilistic'])
            self.assertEqual(summary['models']['trained']['t0']['all']['request_mean_tau'], 3.)
            self.assertTrue(json.loads((root / 'evaluation/COMPLETED.json').read_text())['passed'])
            self.assertEqual((checkpoint / 'model.safetensors').read_bytes(), b'CPU test bytes, never load into a model')


if __name__ == '__main__':
    unittest.main()
