"""Run on each teacher host. A claim starts one fixed, privately owned TP teacher."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import OwnedProcesses, assert_free, identity, source_environment, write
from teachers.protocol import VERSION, model_manifest, source_manifest, runtime_manifest, read_key, publish_complete


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['gpu', 'npu'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--devices', required=True, help='One TP group, e.g. 0,1')
    p.add_argument('--run', type=Path, required=True, help='Fresh shared directory, same absolute mount on trainer')
    p.add_argument('--api-key-file', type=Path, required=True)
    p.add_argument('--bind', default='127.0.0.1', help='Private reachable host IP, or use an SSH tunnel')
    p.add_argument('--port', type=int, default=8395)
    p.add_argument('--native-port', type=int, default=8394)
    p.add_argument('--distributed-port', type=int, default=29300)
    p.add_argument('--target-layer-ids', type=int, nargs='+', default=[6,20,34,48,62])
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-num-seqs', type=int, default=4)
    p.add_argument('--memory-utilization', type=float, default=.8)
    p.add_argument('--lease-seconds', type=int, default=120)
    p.add_argument('--idle-timeout', type=int, default=86400, help='CPU wait for a trainer claim; no accelerator allocated')
    p.add_argument('--request-timeout', type=int, default=900)
    p.add_argument('--dry-run', action='store_true')
    return p


def teacher_command(a, served_name):
    source = ROOT / a.backend / 'source'
    cmd = [sys.executable, str(source/'scripts/launch_vllm.py'), str(a.model),
           '--target-layer-ids', *map(str, a.target_layer_ids), '--hidden-states-backend', 'file',
           '--hidden-states-path', str(a.run/'staging')]
    if a.backend == 'gpu':
        cmd += ['--provenance-dir', str(a.run/'teacher_provenance')]
    cmd += ['--', '--host', '127.0.0.1', '--port', str(a.native_port),
            '--served-model-name', served_name, '--tensor-parallel-size', str(len(a.cards)),
            '--dtype', 'bfloat16', '--enforce-eager', '--no-enable-chunked-prefill',
            '--gpu-memory-utilization', str(a.memory_utilization), '--max-num-seqs', str(a.max_num_seqs),
            '--max-model-len', str(a.max_model_len), '--max-num-batched-tokens', str(a.max_model_len)]
    return cmd


class Engine:
    def __init__(self, a, contract):
        self.a, self.contract = a, contract
        self.lock = threading.RLock()
        self.phase, self.run_id, self.error = 'idle', None, None
        self.last_seen, self.active, self.success, self.failures = time.monotonic(), 0, 0, 0
        self.owner, self.worker = None, None
        self.finished = threading.Event()
        self.key = read_key(a.api_key_file)
        self.log_lock = threading.Lock()
        self.probe_sha256 = None
        self.guardian = None

    def snapshot(self):
        with self.lock:
            return dict(phase=self.phase, run_id=self.run_id, active=self.active,
                        success=self.success, failures=self.failures, error=self.error,
                        probe_sha256=self.probe_sha256)

    def claim(self, request):
        if set(request) != {'run_id', 'contract', 'served_model'}:
            raise ValueError('Invalid teacher claim')
        if request['contract'] != self.contract:
            raise ValueError('Target bytes, source, feature layers, backend, or shared mount differ')
        run_id = request['run_id']
        if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
            raise ValueError('Invalid run ID')
        with self.lock:
            if self.run_id is not None:
                if self.run_id == run_id and self.phase in ['starting', 'ready']:
                    self.last_seen = time.monotonic()
                    return self.snapshot()  # Recover an ambiguous HTTP claim without starting twice.
                raise ValueError('Teacher already belongs to another or completed run')
            self.run_id, self.phase, self.last_seen = run_id, 'starting', time.monotonic()
            probe = os.urandom(64*1024)
            with (self.a.run/'SHARED_PROBE.bin').open('xb') as f:
                f.write(probe); f.flush(); os.fsync(f.fileno())
            self.probe_sha256 = hashlib.sha256(probe).hexdigest()
            write(self.a.run/'CLAIM.json', {'run_id': run_id, 'contract': self.contract,
                                         'served_model': request['served_model']})
            self.worker = threading.Thread(target=self.start_native, args=(request['served_model'],), daemon=True)
            self.worker.start()
            return self.snapshot()

    def touch(self, run_id):
        with self.lock:
            if run_id != self.run_id or self.phase not in ['starting', 'ready']:
                raise ValueError('Teacher lease is not active for this run')
            self.last_seen = time.monotonic()
            return self.snapshot()

    def start_native(self, served_name):
        try:
            if model_manifest(self.a.model) != self.contract['target'] or source_manifest(self.a.backend) != self.contract['source']:
                raise ValueError('Target/source bytes changed while waiting for the trainer')
            assert_free(self.a.backend, self.a.cards)
            source = ROOT/self.a.backend/'source'
            env = source_environment(source)
            env['CUDA_VISIBLE_DEVICES' if self.a.backend == 'gpu' else 'ASCEND_RT_VISIBLE_DEVICES'] = self.a.devices
            env['VLLM_PORT'] = str(self.a.distributed_port)
            env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'npu/scripts') if self.a.backend == 'npu' else str(ROOT),
                                               env.get('PYTHONPATH', '')])
            owned = self.a.run/'owned'; owned.mkdir()
            with self.lock:
                if self.phase != 'starting':
                    return
                self.owner = OwnedProcesses(owned)
                cmd = teacher_command(self.a, served_name)
                write(self.a.run/'teacher_provenance.json', {'argv': cmd, 'source': self.contract['source'],
                      'target': self.contract['target'], 'wrapper_consumed_provenance': self.a.backend == 'npu'})
                proc = self.owner.start('teacher', cmd, env, source)
            deadline = time.monotonic()+1200
            while time.monotonic() < deadline:
                with self.lock:
                    if self.phase != 'starting':
                        return
                if proc.poll() is not None:
                    raise RuntimeError('Native teacher exited before readiness')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.a.native_port}/health', timeout=2) as r:
                        if r.status == 200:
                            with self.lock:
                                if self.phase == 'starting':
                                    self.phase = 'ready'
                                    write(self.a.run/'READY.json', self.snapshot())
                            return
                except OSError:
                    pass
                time.sleep(.5)
            raise TimeoutError('Native teacher startup')
        except BaseException:
            self.error = 'Native teacher startup failed; see remote failure.json'
            write(self.a.run/'failure.json', {'traceback': traceback.format_exc()})
            self.stop('failed', cleanup=False)

    def stop(self, reason, cleanup=False):
        with self.lock:
            if self.phase in ['stopping', 'released', 'expired', 'failed']:
                return
            self.phase = 'stopping'
        deadline = time.monotonic()+self.a.request_timeout+10
        while self.active and time.monotonic() < deadline:
            time.sleep(.05)
        passed, error = True, None
        try:
            if self.owner:
                self.owner.close()
            if self.active:
                raise RuntimeError('Requests did not drain; preserve feature files')
            if (source_manifest(self.a.backend) != self.contract['source']
                    or model_manifest(self.a.model) != self.contract['target']):
                raise ValueError('Remote source or target bytes changed during the run')
            if cleanup:
                marker = json.loads((self.a.run/'OWNER.json').read_text())
                if marker['run'] != str(self.a.run):
                    raise ValueError('Remote cache owner marker changed')
                for name in ['staging', 'ready']:
                    folder = self.a.run/name
                    if folder.is_symlink():
                        raise ValueError('Remote cache became a symlink')
                    shutil.rmtree(folder)
        except BaseException as exc:
            passed, error = False, type(exc).__name__
            write(self.a.run/'cleanup_failure.json', {'traceback': traceback.format_exc()})
        with self.lock:
            self.phase = reason
            write(self.a.run/'CLEANUP.json', {'passed': passed, 'run_id': self.run_id,
                  'reason': reason, 'owned_teacher_released': passed, 'error': error,
                  'cache_removed': cleanup and passed, 'cache_preserved_on_expiry': not cleanup,
                  'utc_epoch': time.time()})
        self.finished.set()

    def watchdog(self):
        initial = time.monotonic()
        while not self.finished.wait(.5):
            with self.lock:
                expired = self.run_id is not None and time.monotonic()-self.last_seen > self.a.lease_seconds
                idle = self.run_id is None and time.monotonic()-initial > self.a.idle_timeout
                dead = self.phase == 'ready' and self.owner and any(p.poll() is not None for p, _ in self.owner.processes)
                dead_guardian = self.guardian is not None and self.guardian.poll() is not None
            if expired or idle or dead or dead_guardian:
                self.stop('failed' if dead or dead_guardian else 'expired', cleanup=False)


def make_server(address, engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def respond(self, code, body):
            data = json.dumps(body, separators=(',', ':')).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+engine.key):
                self.respond(401, {'error': 'Teacher authentication required'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 8*1024**2:
                    raise ValueError('Invalid request length')
                payload = self.rfile.read(length)
                body = json.loads(payload) if payload else {}
                if self.command == 'GET' and self.path == '/__teacher/info':
                    self.respond(200, {'contract': engine.contract, 'lease_seconds': engine.a.lease_seconds,
                                      'max_model_len': engine.a.max_model_len, 'status': engine.snapshot()}); return
                if self.command == 'POST' and self.path == '/__teacher/claim':
                    self.respond(200, engine.claim(body)); return
                if self.command == 'POST' and self.path == '/__teacher/heartbeat':
                    self.respond(200, engine.touch(body.get('run_id'))); return
                if self.command == 'POST' and self.path == '/__teacher/release':
                    if body.get('run_id') != engine.run_id or engine.run_id is None:
                        raise ValueError('Release does not own this teacher')
                    cleanup = body.get('all_local_consumers_stopped') is True
                    threading.Thread(target=engine.stop, args=('released', cleanup), daemon=True).start()
                    self.respond(202, {'releasing': True}); return
                if self.command == 'GET' and self.path in ['/health', '/v1/models']:
                    if engine.snapshot()['phase'] != 'ready':
                        self.respond(503, {'error': 'Teacher not ready'}); return
                    with urllib.request.urlopen(f'http://127.0.0.1:{engine.a.native_port}'+self.path, timeout=10) as r:
                        self.respond(r.status, json.loads(r.read())); return
                if self.command != 'POST' or self.path != '/v1/completions':
                    self.respond(404, {'error': 'Only token-ID hidden-state completions are supported'}); return
                with engine.lock:
                    if engine.phase != 'ready' or time.monotonic()-engine.last_seen > engine.a.lease_seconds:
                        self.respond(503, {'error': 'Teacher lease not ready'}); return
                    engine.active += 1
                code, publication = 502, None
                try:
                    tokens = body.get('prompt')
                    if (not isinstance(tokens, list) or not tokens or len(tokens) > engine.a.max_model_len
                            or any(type(x) is not int or x < 0 for x in tokens)
                            or body.get('max_tokens') != 1):
                        raise ValueError('Expected bounded token-ID prompt and max_tokens=1')
                    req = urllib.request.Request(f'http://127.0.0.1:{engine.a.native_port}'+self.path,
                          data=payload, headers={'Content-Type': 'application/json'}, method='POST')
                    with urllib.request.urlopen(req, timeout=engine.a.request_timeout) as r:
                        result = json.loads(r.read(8*1024**2))
                    choices = result.get('choices') or []
                    if not choices or choices[0].get('prompt_token_ids') != tokens:
                        raise ValueError('Native teacher response changed input token IDs')
                    params = result.get('kv_transfer_params') or {}
                    handle = params.get('hidden_states_path') or params.get('handle')
                    if not handle:
                        raise ValueError('Native teacher returned no hidden-state file')
                    path, publication = publish_complete(handle, engine.a.run/'staging', engine.a.run/'ready',
                         len(engine.a.target_layer_ids)+1, engine.contract['hidden_size'], tokens,
                         timeout=engine.a.request_timeout)
                    params.pop('handle', None); params['hidden_states_path'] = path
                    result['kv_transfer_params'] = params
                    self.respond(200, result); code = 200
                finally:
                    with engine.lock:
                        engine.active -= 1
                        if code == 200: engine.success += 1
                        else: engine.failures += 1
                    with engine.log_lock, (engine.a.run/'routing.jsonl').open('a') as f:
                        f.write(json.dumps({'http_status': code, 'input_sha256': hashlib.sha256(payload).hexdigest(),
                                           'publication': publication, 'utc_epoch': time.time()})+'\n')
            except Exception as exc:
                try:
                    self.respond(502, {'error': type(exc).__name__, 'detail': 'See teacher host evidence'})
                except OSError:
                    pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128
    return Server(address, Handler)


def main():
    a = parser().parse_args()
    a.cards = [int(x) for x in a.devices.split(',')]
    if (not a.cards or min(a.cards) < 0 or len(set(a.cards)) != len(a.cards)
            or a.lease_seconds < 15 or a.idle_timeout < 1 or a.max_num_seqs < 1
            or a.max_model_len < 2 or not 0 < a.memory_utilization < 1
            or len({a.port, a.native_port, a.distributed_port}) != 3
            or any(not 1 <= p <= 65535 for p in [a.port, a.native_port, a.distributed_port])):
        raise ValueError('Invalid device, lease, capacity, or port configuration')
    a.model, a.run = a.model.resolve(), a.run.resolve()
    if a.run == a.model or a.model.is_relative_to(a.run) or a.run.is_relative_to(a.model):
        raise ValueError('Remote run must be separate from model weights')
    if a.dry_run:
        print(json.dumps({'command': teacher_command(a, str(a.model)), 'accelerator_starts_only_after_claim': True,
                          'shared_root': str(a.run), 'backend': a.backend}, indent=2)); return
    if os.name != 'posix' or not Path('/proc').exists():
        raise RuntimeError('Remote teacher execution requires Linux; --dry-run is portable')
    read_key(a.api_key_file)
    if a.run.exists():
        raise ValueError('Remote teacher run must be a new directory; never restart/overwrite')
    for port in [a.native_port, a.distributed_port]:
        with socket.socket() as sock: sock.bind(('127.0.0.1', port))
    raw = json.loads((a.model/'config.json').read_text()); target = raw.get('text_config', raw)
    if any(i < 0 or i >= target['num_hidden_layers'] for i in a.target_layer_ids):
        raise ValueError('Invalid target feature layers')
    if len(set(a.target_layer_ids)) != len(a.target_layer_ids):
        raise ValueError('Duplicate target feature layers')
    contract = {'protocol': VERSION, 'backend': a.backend, 'target': model_manifest(a.model),
                'source': source_manifest(a.backend), 'target_layer_ids': a.target_layer_ids,
                'runtime': runtime_manifest(a.backend), 'hidden_size': target['hidden_size'], 'shared_root': str(a.run)}
    a.run.mkdir(parents=True)
    for name in ['staging', 'ready']: (a.run/name).mkdir()
    write(a.run/'OWNER.json', {'run': str(a.run), 'identity': identity(os.getpid())})
    write(a.run/'CONTRACT.json', contract)
    import importlib.metadata
    versions = {}
    for name in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    write(a.run/'environment.json', versions)
    guardian_log = (a.run/'guardian.log').open('x')
    guardian = subprocess.Popen([sys.executable, str(ROOT/'teachers/guardian.py'), '--run', str(a.run)],
                               stdout=guardian_log, stderr=subprocess.STDOUT, start_new_session=True)
    write(a.run/'guardian_launch_identity.json', identity(guardian.pid))
    engine = Engine(a, contract)
    engine.guardian = guardian
    server = make_server((a.bind, a.port), engine)
    threading.Thread(target=engine.watchdog, daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: engine.finished.set())
    try:
        engine.finished.wait()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop('expired', cleanup=False)
        server.shutdown(); server.server_close()
        try:
            if json.loads((a.run/'CLEANUP.json').read_text())['passed']:
                guardian.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Remote guardian did not acknowledge cleanup; preserve the run')
        finally: guardian_log.close()
    if engine.snapshot()['phase'] == 'failed' or not json.loads((a.run/'CLEANUP.json').read_text())['passed']:
        raise SystemExit(1)


if __name__ == '__main__': main()
