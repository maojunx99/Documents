"""Publication orchestration helpers; no accelerator imports or global process kills."""
from pathlib import Path
import json
import os
import signal
import subprocess
import threading
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, starttime=fields[19], ppid=int(fields[1]), pgid=int(fields[2]), session=int(fields[3]),
                    state=fields[0], cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(expected):
    current = identity(expected['pid'])
    return current if current and current['state'] != 'Z' and all(
        current[k] == expected[k] for k in ['starttime', 'pgid', 'cmdline']) else None


def children_of(parent):
    """Snapshot descendants by PPID before cleanup; never discover by process name."""
    current = identity(parent['pid'])
    if current and any(current[k] != parent[k] for k in ['starttime', 'pgid', 'cmdline']):
        return []  # Reused leader PID must never authorize cleaning a new session.
    found = [parent]
    pending = {parent['pid']}
    candidates = [identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    # start_new_session gives this invocation a private session. This also retains
    # ownership of children orphaned when a launcher fails before cleanup.
    found.extend(p for p in candidates if p and p['pid'] != parent['pid']
                 and p['session'] == parent['pid'] and int(p['starttime']) >= int(parent['starttime']))
    pending.update(p['pid'] for p in found)
    while pending:
        children = [p for p in candidates if p and p['ppid'] in pending and p['pid'] not in {q['pid'] for q in found}]
        found.extend(children)
        pending = {p['pid'] for p in children}
    return found


class OwnedProcesses:
    def __init__(self, run):
        self.run = Path(run)
        self.processes = []
        self.logs = []
        self.lock = threading.RLock()
        self.closing = False

    def start(self, name, argv, env, cwd=None):
        with self.lock:
            if self.closing:
                raise RuntimeError('This process owner is closing')
            return self._start(name, argv, env, cwd)

    def _start(self, name, argv, env, cwd=None):
        log = (self.run / (name + '.log')).open('x')
        self.logs.append(log)
        p = subprocess.Popen([str(x) for x in argv], env=env, cwd=cwd, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        ident = identity(p.pid)
        if ident is None:
            p.wait()
            raise RuntimeError(f'{name} exited before its identity could be recorded')
        self.processes.append((p, ident))
        write(self.run / (name + '_identity.json'), ident)
        write(self.run / (name + '_command.json'), {'argv': [str(x) for x in argv]})
        return p

    def run_checked(self, name, argv, env, cwd=None):
        p = self.start(name, argv, env, cwd)
        if p.wait() != 0:
            raise RuntimeError(f'{name} failed; inspect {self.run / (name + ".log")}')

    def close(self):
        with self.lock:
            if self.closing:
                return
            self.closing = True
        result = []
        for p, ident in reversed(self.processes):
            members = children_of(ident)
            for member in reversed(members):
                if same(member):
                    try:
                        os.kill(member['pid'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 30
            while any(same(x) for x in members) and time.monotonic() < deadline:
                time.sleep(.2)
            for member in reversed(members):
                if same(member):
                    try:
                        os.kill(member['pid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            result.append({'leader': ident, 'members': members, 'remaining': [x for x in members if same(x)]})
        for log in self.logs:
            log.close()
        write(self.run / 'cleanup.json', result)
        if any(x['remaining'] for x in result):
            raise RuntimeError('Some owned processes remain; see cleanup.json')


def source_environment(source, extra=None):
    source = Path(source).resolve()
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(source / 'src'), str(source / 'hs_connectors/src'),
                                      env.get('PYTHONPATH', '')])
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env.update(extra or {})
    return env


def assert_free(backend, devices):
    if len(devices) != len(set(devices)):
        raise ValueError('A device was assigned more than once')
    if backend == 'gpu':
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        uuids = {int(a.strip()): b.strip() for a, b in (line.split(',', 1) for line in rows.splitlines())}
        used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(',')[0].strip() for line in used.splitlines() if line.strip()}
        if any(uuids[d] in occupied for d in devices):
            raise RuntimeError('A selected GPU has a compute process; choose free devices')
    else:
        import re
        text = subprocess.check_output(['npu-smi', 'info'], text=True)
        for d in devices:
            if not re.search(r'No running processes found in NPU ' + str(d) + r'\s', text):
                raise RuntimeError(f'NPU {d} is occupied or its idle state could not be verified')
