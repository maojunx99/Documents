"""Trainer-side admission, lease renewal, and verified cross-host cleanup."""
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

from common.managed import write
from teachers.protocol import VERSION, digest, model_manifest, source_manifest, runtime_manifest, read_key, tensor_file


def rpc(peer, path, body=None, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(peer['url']+path, data=data, headers={
        'Authorization': 'Bearer '+read_key(peer['api_key_file']), 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read(16*1024**2))


class RemoteTeachers:
    def __init__(self, peers, model, backend, target_layers, folder, startup_timeout=1500, required_context=1):
        self.peers, self.model, self.backend = peers, Path(model), backend
        self.target_layers, self.folder, self.startup_timeout = target_layers, Path(folder), startup_timeout
        self.required_context = required_context
        self.run_id = uuid.uuid4().hex
        self.claimed, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = None
        self.state_lock = threading.Lock()

    def heartbeat(self):
        while not self.stop_event.wait(3):
            with self.state_lock: peers = list(self.claimed)
            for peer in peers:
                if self.stop_event.is_set(): return
                try:
                    state = rpc(peer, '/__teacher/heartbeat', {'run_id': self.run_id}, timeout=5)
                    if state['phase'] not in ['starting', 'ready']:
                        raise RuntimeError('Remote teacher no longer active')
                except Exception as exc:
                    with self.state_lock:
                        self.errors.append({'teacher': peer['name'], 'error': type(exc).__name__})
                    write(self.folder/'heartbeat_failure.json', {'errors': list(self.errors)})
                    return  # Fail closed; no silent loss of training examples.

    def check(self):
        with self.state_lock:
            if self.errors:
                raise RuntimeError('Remote teacher lease/health failed; inspect heartbeat_failure.json')

    def start(self):
        if not self.peers: return []
        self.folder.mkdir(parents=True, exist_ok=False)
        raw = json.loads((self.model/'config.json').read_text()); text = raw.get('text_config', raw)
        base = {'protocol': VERSION, 'backend': self.backend, 'target': model_manifest(self.model),
                'source': source_manifest(self.backend), 'target_layer_ids': self.target_layers,
                'runtime': runtime_manifest(self.backend), 'hidden_size': text['hidden_size']}
        write(self.folder/'identity_contract.json', base)
        self.thread = threading.Thread(target=self.heartbeat, daemon=True); self.thread.start()
        backends = []
        for peer in self.peers:
            expected = dict(base, shared_root=peer['shared_root'])
            info = rpc(peer, '/__teacher/info')
            if info['contract'] != expected:
                raise ValueError('Remote teacher target/source/layers/shared mount differs: '+peer['name'])
            if info['lease_seconds'] < 15:
                raise ValueError('Remote teacher lease is too short')
            if info['max_model_len'] < self.required_context:
                raise ValueError('Remote teacher context is shorter than the training request budget')
            # Mark possible ownership BEFORE the HTTP mutation: an ambiguous response must still be cleaned up.
            with self.state_lock: self.claimed.append(peer)
            request = {'run_id': self.run_id, 'contract': expected, 'served_model': str(self.model)}
            reply = rpc(peer, '/__teacher/claim', request)
            deadline = time.monotonic()+self.startup_timeout
            while reply['phase'] != 'ready':
                self.check()
                if reply['phase'] != 'starting' or time.monotonic() > deadline:
                    raise RuntimeError('Remote teacher did not become ready: '+peer['name'])
                time.sleep(.5)
                reply = rpc(peer, '/__teacher/heartbeat', {'run_id': self.run_id})
            shared = Path(peer['shared_root'])
            if digest(shared/'SHARED_PROBE.bin') != reply['probe_sha256']:
                raise ValueError('Trainer cannot read the exact teacher-published shared bytes')
            probes = []
            for n in [32, 256]:
                tokens = [1]*n
                result = rpc(peer, '/v1/completions', {'model': str(self.model), 'prompt': tokens,
                    'max_tokens': 1, 'return_token_ids': True}, timeout=910)
                handle = Path(result['kv_transfer_params']['hidden_states_path'])
                if handle.is_symlink() or handle.parent.resolve() != (shared/'ready').resolve():
                    raise ValueError('Remote probe returned an unowned feature handle')
                receipt = tensor_file(handle, len(self.target_layers)+1, text['hidden_size'], tokens, finite=True)
                if result['choices'][0]['prompt_token_ids'] != tokens:
                    raise ValueError('Remote probe response token IDs mismatch')
                handle.unlink(); probes.append(dict(receipt, finite=True))
            write(self.folder/(peer['name']+'_admission.json'), {'passed': True, 'run_id': self.run_id,
                  'url': peer['url'], 'shared_root': str(shared), 'probes': probes,
                  'target_and_source_bytes_match': True, 'cross_host_flock_used': False,
                  'hidden_state_byte_determinism_claimed': False})
            backends.append(dict(peer, ready_root=str(shared/'ready')))
        write(self.folder/'ADMISSION.json', {'passed': True, 'run_id': self.run_id,
              'teachers': [p['name'] for p in self.peers]})
        return backends

    def close(self, consumers_stopped=True):
        if not self.peers: return
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=10)
        receipts, errors = [], []
        for peer in self.claimed:
            root = Path(peer['shared_root'])
            try:
                claim = json.loads((root/'CLAIM.json').read_text())
                if claim['run_id'] != self.run_id:
                    continue  # A rejected claim never authorizes stopping somebody else's teacher.
                try:
                    rpc(peer, '/__teacher/release', {'run_id': self.run_id,
                        'all_local_consumers_stopped': consumers_stopped})
                except OSError:
                    pass  # Already expired/exited: the shared receipt is authoritative, not a network guess.
                deadline = time.monotonic()+1000
                path = root/'CLEANUP.json'
                while not path.exists() and time.monotonic() < deadline:
                    guardian = root/'GUARDIAN_CLEANUP.json'
                    if guardian.exists() and json.loads(guardian.read_text()).get('owner_exited'):
                        raise RuntimeError('Remote owner died; guardian preserves features for diagnosis')
                    time.sleep(.2)
                receipt = json.loads(path.read_text())
                if (receipt['run_id'] != self.run_id or not receipt['passed']
                        or (consumers_stopped and not receipt['cache_removed'])):
                    raise RuntimeError('Remote cleanup not verified; preserve remaining shared files')
                if (root/'guardian_launch_identity.json').exists():
                    guardian = root/'GUARDIAN_CLEANUP.json'
                    deadline = time.monotonic()+30
                    while not guardian.exists() and time.monotonic() < deadline:
                        time.sleep(.1)
                    if not json.loads(guardian.read_text()).get('passed'):
                        raise RuntimeError('Remote guardian cleanup was not verified')
                receipts.append(dict(teacher=peer['name'], **receipt))
            except Exception as exc:
                errors.append({'teacher': peer['name'], 'error': type(exc).__name__})
        write(self.folder/'CLEANUP.json', {'passed': not errors, 'receipts': receipts, 'errors': errors})
        if errors:
            raise RuntimeError('Remote teacher cleanup was not verified')


def connect(peers, model, backend, target_layers, folder, required_context=1):
    client = RemoteTeachers(peers, model, backend, target_layers, folder, required_context=required_context)
    try:
        backends = client.start()
    except BaseException:
        client.close()
        raise
    return client, backends


def router_environment(folder, local_ports, remote_backends):
    """Only paths to private key files are serialized; never the key values."""
    path = Path(folder)/'ROUTER_BACKENDS.json'
    backends = [{'name': 'local_'+str(p), 'url': f'http://127.0.0.1:{p}'} for p in local_ports]
    backends += remote_backends
    if not backends: raise ValueError('At least one local or remote teacher is required')
    write(path, {'backends': backends})
    return {'TEACHER_BACKENDS_FILE': str(path)}
