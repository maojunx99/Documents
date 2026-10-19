"""CPU checks for the public shell steps; no Docker install or accelerator jobs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npu.prepare_smoke_inputs import select

if os.name == 'nt':
    candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    BASH = str(candidate) if candidate.exists() else None
else:
    BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'Requires Bash for the public shell steps')
class NpuSteps(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='npu_steps_')
        self.root = Path(self.temp.name)
        self.model = self.root / 'target'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'text_config': {
            'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 64}}))
        self.prompts = self.root / 'prompts.jsonl'
        self.prompts.write_text(''.join(json.dumps({'id': str(i), 'messages': [
            {'role': 'user', 'content': f'Question {i}'}]}) + '\n' for i in range(140)))
        executable = Path(sys.executable).as_posix()
        if os.name == 'nt':
            executable = '/' + executable[0].lower() + executable[2:]
        self.env = dict(os.environ, PYTHON=executable, PYTHONUTF8='1',
                        TARGET_MODEL=self.model.as_posix(), RUN_ROOT=self.root.as_posix())
        # Do not inherit optional recipe overrides from the developer's shell.
        for name in ['RUN_DIR', 'REGENERATED_DIR', 'MAX_STEPS', 'FSDP_REPLICAS',
                     'FSDP_SHARD_SIZE', 'TRAIN_DEVICES', 'TEACHER_TP_GROUPS']:
            self.env.pop(name, None)
        self.config = self.root / 'config.sh'
        self.call('03_configure.sh', '--output', str(self.config))
        text = self.config.read_text()
        text = text.replace('INPUT_FILES=(/data/prompts/open_perfectblend.jsonl /data/prompts/nemotron_v2.jsonl)',
                            'INPUT_FILES=(' + shlex.quote(self.prompts.as_posix()) + ')')
        self.config.write_text(text, newline='\n')

    def tearDown(self):
        self.temp.cleanup()

    def call(self, script, *args, ok=True, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        result = subprocess.run([BASH, str(ROOT / 'examples/npu' / script), *args],
                                env=env, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def plan(self, *args, **kwargs):
        return json.loads(self.call('04_run.sh', '--config', str(self.config), '--dry-run',
                                    *args, **kwargs).stdout)

    def test_config_is_not_overwritten(self):
        before = self.config.read_bytes()
        self.call('03_configure.sh', '--output', str(self.config))
        self.assertEqual(self.config.read_bytes(), before)

    def test_host_and_install_dry_run_are_inert(self):
        untouched = self.root / 'not_created'
        result = self.call('01_container.sh', '--dry-run', extra_env={
            'HOST_WORK_ROOT': '/unused/work', 'HOST_MODEL_ROOT': '/unused/models',
            'HOST_PROMPT_ROOT': '/unused/prompts', 'HOST_RUN_ROOT': '/unused/runs',
            'NPU_IDS': '0,1,2,3,4,5,6,7'})
        self.assertIn('docker pull', result.stdout)
        self.assertIn('/workspace/pipeline', result.stdout)
        self.call('01_container.sh', '--dry-run', ok=False, extra_env={'NPU_IDS': '0,0'})
        result = self.call('02_install.sh', '--dry-run', extra_env={'VENV_PATH': untouched.as_posix()})
        self.assertIn('base-stack.txt', result.stdout)
        self.assertFalse(untouched.exists())

    def test_smoke_does_not_change_full_configuration(self):
        before = self.config.read_bytes()
        full = self.plan()
        smoke = self.plan('--smoke')
        again = self.plan()
        self.assertEqual(full['config']['input'], [str(self.prompts)])
        self.assertEqual(full['rank_mesh'], [[0, 1, 2, 3]])
        self.assertEqual(full['config']['generation_max_tokens'], 8192)
        self.assertEqual(full['config']['epochs'], 5)
        self.assertIsNone(full['config']['max_steps'])
        self.assertEqual(smoke['config']['max_steps'], 2)
        self.assertEqual(smoke['config']['generation_max_tokens'], 128)
        self.assertEqual(smoke['config']['train_ratio'], .8)
        self.assertEqual(again['config']['generation_max_tokens'], 8192)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse(Path(smoke['config']['run']).exists())
        self.assertFalse(Path(smoke['config']['input'][0]).exists())

    def test_topology_sampling_reuse_and_existing_run_guards(self):
        plan = self.plan(extra_env={'FSDP_REPLICAS': '2', 'TRAIN_DEVICES': '8,9,10,11,12,13,14,15',
                                    'TEACHER_TP_GROUPS': '2,3', 'TEMPERATURE': '.8', 'TOP_K': '40'})
        self.assertEqual(plan['rank_mesh'], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(plan['config']['temperature'], .8)
        self.assertEqual(plan['config']['top_k'], 40)
        reused = self.plan(extra_env={'REGENERATED_DIR': self.root.as_posix()})
        self.assertIsNone(reused['config']['input'])
        self.assertNotIn('regeneration', reused['commands'])
        self.call('04_run.sh', '--config', str(self.config), '--dry-run', ok=False,
                  extra_env={'RUN_DIR': self.root.as_posix()})


class SmokeSelection(unittest.TestCase):
    def test_selection_preserves_sources_and_rejects_overwrite_or_short_input(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'source.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'messages': []}) + '\n' for i in range(140)))
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            output = folder / 'new'
            selected = select([source], output)
            self.assertEqual(len(selected.read_text().splitlines()), 128)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            with self.assertRaises(FileExistsError):
                select([source], output)
            source.write_text(json.dumps({'messages': []}) + '\n')
            with self.assertRaisesRegex(ValueError, 'found 1'):
                select([source], folder / 'too_short')
            self.assertFalse((folder / 'too_short').exists())


if __name__ == '__main__':
    unittest.main()
