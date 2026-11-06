"""NPU teacher nodes from one config. SSH controls processes; shared storage carries tensors."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import uuid

from common.managed import write
from teachers.client import rpc
from teachers.protocol import digest

ROOT = Path(__file__).resolve().parents[1]


def nodes(rows, default_container, train_devices=()):
    result, used = [], {}
    for row in rows:
        parts = row.split('|')
        if len(parts) not in [3, 4]:
            raise ValueError('Teacher node: HOST|PHYSICAL_CARDS|TP[|EXISTING_CONTAINER]')
        host, cards, tp = (part.strip() for part in parts[:3])
        container = parts[3].strip() if len(parts) == 4 else default_container
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
            raise ValueError('Use local, an IPv4/DNS name, or an SSH alias (optional user@host)')
        if host != 'local' and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', container):
            raise ValueError('Each remote node needs an existing container name')
        cards = [int(x.strip()) for x in cards.split(',')]
        tp = int(tp)
        if tp < 1 or not cards or min(cards) < 0 or len(cards) % tp or len(cards) != len(set(cards)):
            raise ValueError('Cards must be distinct and divide into complete TP groups')
        previous = used.setdefault(host, set())
        if previous.intersection(cards) or (host == 'local' and set(cards).intersection(train_devices)):
            raise ValueError('Teacher card assigned twice or overlaps local training cards')
        previous.update(cards)
        for i in range(0, len(cards), tp):
            result.append(dict(host=host, container=container, cards=cards[i:i+tp]))
    return result


def add_arguments(p):
    p.add_argument('--teacher-node', action='append', default=[], help='HOST|PHYSICAL_CARDS|TP[|EXISTING_CONTAINER]; repeat per host')
    p.add_argument('--teacher-shared-root', type=Path, help='Same writable absolute mount on trainer and teacher containers')
    p.add_argument('--teacher-container', default='dflash2-27b-train', help='Existing remote container; optional per-node override')
    p.add_argument('--teacher-python', help='Python path inside existing remote containers; default --python')
    p.add_argument('--teacher-model', type=Path, help='Target path inside remote containers; default --model; bytes must match')
    p.add_argument('--teacher-ssh-user', default='', help='Optional common SSH username; aliases/user@host take precedence')
    p.add_argument('--teacher-node-port', type=int, default=18395, help='First native HTTP port; reserve 200 ports per TP group')


def validate(a, train):
    result = nodes(a.teacher_node, a.teacher_container, train)
    if result:
        if a.hidden_states_backend != 'shared_file':
            raise ValueError('Automatic shared-disk teachers require --hidden-states-backend shared_file')
        if a.remote_teachers or a.teacher_groups.strip():
            raise ValueError('teacher-node includes local/remote nodes; leave legacy teacher-groups/remote-teachers empty')
        if a.teacher_shared_root is None or not a.teacher_shared_root.is_absolute():
            raise ValueError('teacher-node needs an absolute --teacher-shared-root')
        if a.teacher_node_port < 1024 or a.teacher_node_port+200*(len(result)-1)+199 > 65535:
            raise ValueError('Invalid teacher-node port range')
        if a.router_port in {a.teacher_node_port+200*i+j for i in range(len(result)) for j in [0,1,*range(100,200)]}:
            raise ValueError('Teacher-node ports overlap router port')
        if a.teacher_ssh_user and not re.fullmatch(r'[A-Za-z0-9_.-]+', a.teacher_ssh_user):
            raise ValueError('Invalid SSH username')
    return result


def snapshot(destination):
    """Copy the small published code bundle, never target weights/data or local secrets."""
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    entries = manifest['files']
    if isinstance(entries, list):
        entries = {x['path']: x['sha256'] for x in entries}
    destination.mkdir()
    for name, item in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source manifest path')
        expected = item['sha256'] if isinstance(item, dict) else item
        source = ROOT/relative
        if digest(source) != expected:
            raise ValueError('Published code manifest mismatch: '+name)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError('Shared code snapshot mismatch: '+name)
    shutil.copyfile(ROOT/'SOURCE_MANIFEST.json', destination/'SOURCE_MANIFEST.json')


class Deployment:
    def __init__(self, a, resolved, folder):
        self.a, self.specs, self.folder = a, resolved['teacher_nodes'], Path(folder)
        self.job = a.teacher_shared_root/('teachers_'+uuid.uuid4().hex)
        self.launched = []
        self.host_cards = {}

    def invoke(self, spec, control, action):
        py = self.a.python if spec['host'] == 'local' else (self.a.teacher_python or self.a.python)
        argv = [py, str(self.job/'pipeline/teachers/node_agent.py'), action, '--control', str(control)]
        if spec['host'] == 'local':
            return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=90)
        # SSH is used twice (start/verify-stop), never to read feature tensors.
        host = spec['host']
        if self.a.teacher_ssh_user and '@' not in host:
            host = self.a.teacher_ssh_user+'@'+host
        remote = ('set -e; '
                  'if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; '
                  'if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then . /usr/local/Ascend/nnal/atb/set_env.sh; fi; '
                  'exec '+shlex.join(argv))
        command = shlex.join(['docker', 'exec', spec['container'], 'bash', '-c', remote])
        try:
            return subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                                   '-o', 'ConnectTimeout=12', host, command],
                                  check=True, capture_output=True, text=True, timeout=90)
        except subprocess.CalledProcessError as exc:
            write(control/('dispatch_'+action+'_failure.json'), {'returncode': exc.returncode,
                  'stdout': exc.stdout, 'stderr': exc.stderr})
            raise

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        self.job.mkdir(parents=True, mode=0o755, exist_ok=False)
        snapshot(self.job/'pipeline')
        write(self.folder/'JOB.json', {'shared_job':str(self.job),'source':str(self.job/'pipeline'),
              'native_http_direct':True,'authentication':False,'extra_gateway':False})
        peers = []
        for i, spec in enumerate(self.specs):
            name, port = f'node_{i}', self.a.teacher_node_port+200*i
            control = self.job/(name+'_control')
            control.mkdir()
            py = self.a.python if spec['host'] == 'local' else (self.a.teacher_python or self.a.python)
            model = self.a.model if spec['host'] == 'local' else (self.a.teacher_model or self.a.model)
            service = self.job/name
            command = [py, str(self.job/'pipeline/teachers/launch.py'), '--backend', 'npu',
                '--model', str(model), '--devices', ','.join(map(str, spec['cards'])),
                '--run', str(service), '--hidden-states-path', str(self.a.hidden_cache or self.a.run/'hidden_states'),
                '--served-model-name',str(self.a.model),
                '--bind', '127.0.0.1' if spec['host'] == 'local' else '0.0.0.0',
                '--port', str(port), '--distributed-port', str(port+100),
                '--target-layer-ids', *map(str, self.a.target_layer_ids),
                '--max-model-len', str(max(8192, self.a.packed_tokens+8)),
                '--max-num-seqs', str(self.a.teacher_max_num_seqs),
                '--memory-utilization', str(self.a.teacher_memory_utilization)]
            write(control/'request.json', {'argv': command, 'service_run': str(service), 'spec': spec})
            self.invoke(spec,control,'inspect')
            host_id = json.loads((control/'host.json').read_text())['boot_id']
            used = self.host_cards.setdefault(host_id, set())
            local_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if used.intersection(spec['cards']) or (host_id == local_id and set(spec['cards']).intersection(
                    int(x) for x in self.a.train_devices.split(','))):
                raise ValueError('Aliases refer to overlapping physical teacher/trainer cards on the same host')
            used.update(spec['cards'])
            self.launched.append((spec, control))  # Ambiguous SSH result never authorizes another start.
            result = self.invoke(spec, control, 'start')
            write(control/'dispatch.json', {'stdout': result.stdout, 'stderr': result.stderr})
            if not (control/'identity.json').is_file():
                raise RuntimeError('Teacher launcher did not publish its identity')
            host = spec['host'].split('@')[-1]
            if host == 'local':
                host = '127.0.0.1'
            else:
                # Resolve SSH aliases on the launcher, not inside the remote container.
                info = subprocess.check_output(['ssh', '-G', spec['host']], text=True)
                host = next(line.split(None, 1)[1] for line in info.splitlines() if line.startswith('hostname '))
            peer = dict(name=name, url=f'http://{host}:{port}', shared_root=str(self.a.hidden_cache or self.a.run/'hidden_states'))
            deadline = time.monotonic()+1500
            last_alive_check=0
            while True:
                if time.monotonic()-last_alive_check>15:
                    self.invoke(spec,control,'check');last_alive_check=time.monotonic()
                if (control/'failure.json').exists() or (service/'failure.json').exists():
                    raise RuntimeError('Auto teacher startup failed; inspect '+str(control))
                try:
                    rpc(peer, '/health', timeout=3)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Native teacher readiness: '+name)
                    time.sleep(1)
            mount=json.loads((service/'SHARED_PROBE.json').read_text())
            probe=Path(mount['path'])
            if digest(probe)!=mount['sha256']:
                raise ValueError('Teacher shared filesystem is not the trainer mount')
            probe.unlink()
            remote=json.loads((service/'TARGET_IDENTITY.json').read_text())
            local={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,
                          'sha256':digest(f) if f.stat().st_size<1024**2 else None}
                   for f in self.a.model.iterdir() if f.is_file()}
            if remote!=local:
                raise ValueError('Teacher target file/config identity differs; use the same shared target mount')
            peers.append(peer)
        write(self.folder/'PEERS.json', {'teachers': peers})
        return peers

    def close(self, consumers_stopped=True):
        errors = []
        for spec, control in reversed(self.launched):
            try:
                self.invoke(spec, control, 'stop')
            except Exception as exc:
                errors.append({'control': str(control), 'error': type(exc).__name__})
        passed = not errors
        write(self.folder/'CLEANUP.json', {'passed':passed,'errors':errors,
              'consumers_stopped':consumers_stopped,'shared_job':str(self.job),
              'native_http_direct':True,'authentication':False})
        if errors:
            raise RuntimeError('Auto teacher process cleanup incomplete; inspect shared control identities')


def start(a, resolved, folder):
    deployment = Deployment(a, resolved, folder)
    try:
        peers = deployment.start()
    except BaseException:
        deployment.close()
        raise
    return deployment, peers
