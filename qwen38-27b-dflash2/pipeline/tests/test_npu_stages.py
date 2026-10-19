"""Stage handoff and refusal tests use CPU stand-ins, never actual native training."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from npu.recipe import parser, validate, command_plan
from npu import stages


class StageHandoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'target'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'text_config': {
            'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 64}}))
        self.prompts = self.root / 'prompts.jsonl'
        self.prompts.write_text('{"messages": []}\n')
        self.source = self.root / 'implementation.py'
        self.source.write_text('original')
        self.run = self.root / 'run'
        # Change only this module's OS view; do not alter pathlib's Windows/POSIX selection.
        posix_view = types.SimpleNamespace(**vars(os))
        posix_view.name = 'posix'
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(stages, 'os', posix_view))
        self.stack.enter_context(patch.object(stages, 'protected_files', side_effect=lambda _: {str(self.source): stages.sha(self.source)}))
        for name in ['preflight', 'regenerate', 'prepare', 'train']:
            self.stack.enter_context(patch.object(stages.Stage, name, self.cpu_stage))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    @staticmethod
    def cpu_stage(stage):
        product = stage.run / (stage.a.stage + '_test_artifact.json')
        stages.write(product, {'stage': stage.a.stage, 'test_only': True})
        stage.products = [product]
        if stage.a.stage == 'train':
            stages.write(stage.run / 'TRAINING_VERIFICATION.json', {'completed_epochs': [0], 'test_only': True})
            stage.products.append(stage.run / 'TRAINING_VERIFICATION.json')

    def args(self, name, *extra):
        a = parser().parse_args(['--model', str(self.model), '--run', str(self.run),
                                '--input', str(self.prompts), '--stage', name, *extra])
        resolved = validate(a)
        return a, resolved, command_plan(a, resolved)

    def call(self, name, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            stages.execute(*self.args(name, *extra))

    def test_independent_phase_calls_reuse_verified_outputs(self):
        for name in ['preflight', 'regenerate', 'prepare', 'train']:
            self.call(name)
            receipt = json.loads((self.run / 'stages' / name / 'COMPLETED.json').read_text())
            self.assertTrue(receipt['passed'])
            stages.check_products(receipt)
        self.call('regenerate')  # Completed stages are verified and skipped, not rerun.
        self.assertTrue(json.loads((self.run / 'COMPLETION_VERIFICATION.json').read_text())['passed'])
        self.assertFalse((self.run / '.stage.lock').exists())

    def test_missing_prerequisite_and_configuration_changes_are_rejected(self):
        self.call('preflight')
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            self.call('prepare')
        with self.assertRaisesRegex(ValueError, 'unchanged training configuration'):
            self.call('regenerate', '--lr', '.001')
        self.assertFalse((self.run / 'stages/regenerate').exists())

    def test_artifact_source_and_raw_input_tampering_fail_before_work(self):
        self.call('preflight')
        product = self.run / 'preflight_test_artifact.json'
        original = product.read_bytes()
        product.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Stage artifact changed'):
            self.call('regenerate')
        product.write_bytes(original)
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Protected implementation'):
            self.call('regenerate')
        self.source.write_text('original')
        self.prompts.write_text('{"messages": [1]}')
        with self.assertRaisesRegex(ValueError, 'raw input changed'):
            self.call('regenerate')

    def test_failed_attempt_and_existing_lock_are_not_overwritten(self):
        self.call('preflight')
        folder = self.run / 'stages/regenerate'
        folder.mkdir()
        failure = folder / 'failure.json'
        failure.write_text('original failure evidence')
        with self.assertRaisesRegex(ValueError, 'unfinished/failed'):
            self.call('regenerate')
        self.assertEqual(failure.read_text(), 'original failure evidence')
        # An unrelated/stale lock is never guessed away.
        lock = self.run / '.stage.lock'
        lock.write_text('do not remove')
        with self.assertRaises(FileExistsError):
            with stages.locked(self.run, 'evaluate'):
                pass
        self.assertEqual(lock.read_text(), 'do not remove')

    def test_evaluation_handoff_does_not_require_archived_raw_training_data(self):
        for name in ['preflight', 'regenerate', 'prepare', 'train']:
            self.call(name)
        self.prompts.unlink()
        with patch('evaluation.run.run_evaluation', return_value={'test_only': True}) as evaluator:
            self.call('evaluate')
        evaluator.assert_called_once()
        self.assertFalse((self.run / '.stage.lock').exists())


if __name__ == '__main__':
    unittest.main()
