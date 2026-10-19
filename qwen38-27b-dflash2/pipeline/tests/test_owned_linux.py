"""Real Linux processes: exec transition, orphan cleanup, and unrelated-process preservation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from managed import OwnedProcesses,identity,same


@unittest.skipUnless(sys.platform=='linux','Requires Linux /proc process identities')
class OwnedCleanup(unittest.TestCase):
    def test_exec_orphan_and_unrelated_sentinel(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);owner=OwnedProcesses(root)
            sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
            sentinel_id=identity(sentinel.pid)
            try:
                # Parent exits after spawning a same-session child. The orphan remains owned.
                child_id_file=root/'orphan.json'
                parent_code=('import subprocess,sys,pathlib,json;'
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);'
                    f'pathlib.Path({str(child_id_file)!r}).write_text(json.dumps({{"pid":p.pid}}))')
                parent=owner.start('orphan_parent',[sys.executable,'-c',parent_code],os.environ.copy())
                parent.wait(timeout=10)
                child=identity(json.loads(child_id_file.read_text())['pid'])
                # Delayed exec guarantees initial recorded argv differs from final argv.
                process=owner.start('exec',[sys.executable,'-c',
                    'import os,sys,time;time.sleep(.3);os.execv(sys.executable,[sys.executable,"-c","import time;time.sleep(120)"])'],os.environ.copy())
                initial=owner.processes[-1][1]
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    current=identity(process.pid)
                    if current and current['cmdline']!=initial['cmdline']:break
                    time.sleep(.05)
                self.assertNotEqual(current['cmdline'],initial['cmdline'])
                self.assertEqual(current['starttime'],initial['starttime'])
                owner.close()
                self.assertIsNone(same(current));self.assertIsNone(same(child))
                self.assertIsNotNone(same(sentinel_id))
                self.assertTrue(all(not row['remaining'] for row in json.loads((root/'cleanup.json').read_text())))
            finally:
                owner.close()
                sentinel.terminate();sentinel.wait(timeout=10)


if __name__=='__main__':unittest.main()
