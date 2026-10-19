"""Preparation audits must not walk formatted Tensor scalars or alter saved data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from npu import stages


class SavedDatasetView:
    """CPU stand-in for Dataset's non-mutating formatting views."""
    def __init__(self, rows, format_type='torch'):
        self.rows = rows
        self.format_type = format_type

    def with_format(self, format_type):
        return SavedDatasetView(self.rows, format_type)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        if self.format_type == 'torch':
            raise AssertionError('Audit entered the per-token Tensor scalar path')
        return iter(self.rows)


class PreparedAudit(unittest.TestCase):
    def audit(self, root, rows, format_type='torch', train_ratio=.5):
        saved = SavedDatasetView(rows, format_type)
        loader = Mock(return_value=saved)
        stage = SimpleNamespace(run=root, commands={'prepare': ['native-prepare']},
                                checked=Mock(), products=[],
                                a=SimpleNamespace(sample_max_length=4096,
                                                  train_ratio=train_ratio,
                                                  replicas=1, shard_size=1))
        with patch.dict(sys.modules, {'datasets': SimpleNamespace(load_from_disk=loader)}):
            stages.Stage.prepare(stage)
        loader.assert_called_once_with(str(root / 'prepared'))
        stage.checked.assert_called_once_with('prepare', ['native-prepare'])
        self.assertEqual(saved.format_type, format_type)
        return json.loads((root / 'prepared_audit.json').read_text())

    def test_exact_counts_and_saved_format_are_preserved(self):
        rows = [{'input_ids': [7]*4096, 'loss_mask': [0]*1024+[1]*3072},
                {'input_ids': [1, 2, 3], 'loss_mask': [0, 1, 1]},
                {'input_ids': [4, 5], 'loss_mask': [0, 1]},
                {'input_ids': [6], 'loss_mask': [1]}]
        for format_type in ['torch', None]:
            with self.subTest(format_type=format_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prepared = root / 'prepared'
                prepared.mkdir()
                state = prepared / 'state.json'
                state.write_text(json.dumps({'_format_type': format_type}))
                before = state.read_bytes()
                result = self.audit(root, rows, format_type)
                self.assertEqual(result, {'rows': 4, 'input_tokens': 4102,
                                          'supervised_tokens': 3076, 'max_length': 4096})
                self.assertEqual(state.read_bytes(), before)
                self.assertEqual(json.loads((root / 'prepared_fingerprints.json').read_text()),
                                 {'state.json': hashlib.sha256(before).hexdigest()})

    def test_invalid_samples_still_fail(self):
        bad_rows = [{'input_ids': [], 'loss_mask': []},
                    {'input_ids': [1]*4097, 'loss_mask': [1]*4097},
                    {'input_ids': [1, 2], 'loss_mask': [1]},
                    {'input_ids': [1, 2], 'loss_mask': [0, 0]}]
        for row in bad_rows:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(AssertionError):
                self.audit(Path(tmp), [row])

    def test_insufficient_split_still_fails(self):
        row = {'input_ids': [1], 'loss_mask': [1]}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'too small'):
            self.audit(Path(tmp), [row], train_ratio=.99)
