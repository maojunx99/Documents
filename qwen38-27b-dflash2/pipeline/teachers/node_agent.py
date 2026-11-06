"""Short-lived launcher inside an existing container; all native process operations are identity-scoped."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import identity, same, children_of, write


def execute(action, control):
    control = control.resolve()
    request = json.loads((control/'request.json').read_text())
    ident_path = control/'identity.json'
    if action == 'inspect':
        write(control/'host.json',{'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        return
    if action == 'start':
        # No retries, even after an ambiguous SSH disconnect or an early process exit.
        with (control/'STARTED').open('x') as f:
            f.write(str(os.getpid()))
        write(control/'host.json', {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        command = request['argv']
        with (control/'service.log').open('x') as log:
            p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        ident = identity(p.pid)
        if not ident:
            raise RuntimeError('Native teacher exited before identity capture')
        write(ident_path, ident)
        print(json.dumps({'started': True, 'identity': ident}))
        return
    if action == 'check':
        ident=json.loads(ident_path.read_text());members=children_of(ident)
        if not any(same(x) for x in members):raise RuntimeError('Native teacher exited; inspect service.log')
        print(json.dumps({'alive':True}));return
    if not ident_path.exists():
        if (control/'STARTED').exists():
            raise RuntimeError('Ambiguous launch without identity; preserve for manual inspection')
        write(control/'STOPPED.json', {'passed': True, 'not_started': True})
        return
    ident = json.loads(ident_path.read_text())
    members = children_of(ident)
    for member in reversed(members):
        if same(member):
            try: os.kill(member['pid'],signal.SIGTERM)
            except ProcessLookupError: pass
    deadline = time.monotonic()+30
    while any(same(x) for x in members) and time.monotonic()<deadline:
        time.sleep(.2)
    for member in reversed(members):
        if same(member):
            try:os.kill(member['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    deadline=time.monotonic()+10
    while any(same(x) for x in members) and time.monotonic()<deadline:time.sleep(.1)
    remaining=[x for x in members if same(x)]
    write(control/'STOPPED.json',{'passed':not remaining,'leader':ident,'members':members,'remaining':remaining})
    if remaining:raise RuntimeError('Some owned native teacher processes remain')
    print(json.dumps({'stopped': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'stop', 'check', 'inspect'])
    p.add_argument('--control', type=Path, required=True)
    a = p.parse_args()
    try:
        execute(a.action, a.control)
    except BaseException:
        write(a.control/'failure.json', {'traceback': traceback.format_exc(), 'action': a.action})
        raise
