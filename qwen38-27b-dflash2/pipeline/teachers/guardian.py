"""Independent CPU guardian: preserve feature files if the gateway owner dies."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import children_of, identity, same, write


def guard(run):
    expected = json.loads((run/'OWNER.json').read_text())['identity']
    write(run/'guardian_identity.json', identity(os.getpid()))
    while same(expected):
        complete = run/'CLEANUP.json'
        if complete.exists() and json.loads(complete.read_text()).get('passed'):
            write(run/'GUARDIAN_CLEANUP.json', {'passed': True, 'canonical_cleanup_verified': True,
                                               'cache_deleted': False})
            return
        time.sleep(.5)
    identities = []
    for path in (run/'owned').glob('*_identity.json'):
        identities.extend(children_of(json.loads(path.read_text())))
    identities = list({(x['pid'], x['starttime']): x for x in identities}.values())
    for member in reversed(identities):
        if same(member):
            try: os.kill(member['pid'], signal.SIGTERM)
            except ProcessLookupError: pass
    deadline = time.monotonic()+30
    while any(same(x) for x in identities) and time.monotonic() < deadline:
        time.sleep(.1)
    for member in reversed(identities):
        if same(member):
            try: os.kill(member['pid'], signal.SIGKILL)
            except ProcessLookupError: pass
    deadline = time.monotonic()+5
    while any(same(x) for x in identities) and time.monotonic() < deadline:
        time.sleep(.1)
    remaining = [x for x in identities if same(x)]
    write(run/'GUARDIAN_CLEANUP.json', {'passed': not remaining, 'owner_exited': True,
          'identities': identities, 'remaining': remaining, 'cache_deleted': False,
          'limitation': 'Preserve staging/ready files until every trainer consumer has exited.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    guard(p.parse_args().run.resolve())
