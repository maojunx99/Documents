# Publication changes: configurable router ports and output path.
# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""HTTP router over local ports or explicitly admitted authenticated remote teachers.

Adapted from the 4B-line TeacherRouter (dynamic_teacher_pool.py) but simplified:
No dynamic membership: least-active dispatch to a fixed set of TP replicas.
Remote admission/leases and shared-file publication are managed by teachers/."""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teachers.protocol import read_key

PORT = int(os.environ.get('TEACHER_ROUTER_PORT','8250'))
if os.environ.get('TEACHER_BACKENDS_FILE'):
    items = json.loads(Path(os.environ['TEACHER_BACKENDS_FILE']).read_text())['backends']
else:
    items = [{'name': 'local_'+x, 'url': 'http://127.0.0.1:'+x}
             for x in os.environ.get('TEACHER_BACKEND_PORTS','8260,8261,8262,8263').split(',') if x]
if not items:
    raise ValueError('Teacher router needs at least one backend')
BACKENDS = dict(enumerate(items))
LOG = Path(os.environ.get('TEACHER_ROUTING_LOG','router_routing.jsonl'))
LOG.parent.mkdir(parents=True, exist_ok=True)


def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Router:
    def __init__(self, backends):
        self.lock = threading.Lock()
        self.members = backends
        self.active = {g: 0 for g in backends}
        self.sequence = 0

    def pick(self):
        with self.lock:
            candidates = list(self.members)
            minimum = min(self.active[g] for g in candidates)
            available = [g for g in candidates if self.active[g] == minimum]
            gpu = available[self.sequence % len(available)]
            self.sequence += 1
            seq = self.sequence
            self.active[gpu] += 1
            return gpu, self.members[gpu], seq

    def done(self, gpu, seq, code):
        with self.lock:
            self.active[gpu] -= 1
            with LOG.open('a') as f:
                f.write(json.dumps(dict(time=now(), sequence=seq, gpu=gpu, http_status=code)) + '\n')


router = Router(BACKENDS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        size = int(self.headers.get('Content-Length', '0'))
        payload = self.rfile.read(size) if size else None
        gpu, backend, seq = router.pick()
        code = 502
        try:
            headers = {'Content-Type': 'application/json'}
            if backend.get('api_key_file'):
                headers['Authorization'] = 'Bearer '+read_key(backend['api_key_file'])
            req = urllib.request.Request(backend['url'] + self.path, data=payload,
                                         method=self.command, headers=headers)
            try:
                response = urllib.request.urlopen(req, timeout=610)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                content = response.read()
                code = response.status
                if code == 200 and self.command == 'POST' and backend.get('ready_root'):
                    body = json.loads(content)
                    handle = Path(body['kv_transfer_params']['hidden_states_path'])
                    if (handle.is_symlink() or handle.parent.resolve() != Path(backend['ready_root']).resolve()
                            or handle.suffix != '.safetensors' or not handle.is_file()
                            or Path(str(handle)+'.lock').exists()):
                        raise ValueError('Remote teacher returned an unpublished or unowned feature file')
                self.send_response(code)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception as exc:
            code = 502
            try:
                self.send_error(502, type(exc).__name__)
            except OSError:
                pass
        finally:
            router.done(gpu, seq, code)


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


if __name__ == '__main__':
    server = Server(('127.0.0.1', PORT), Handler)
    print(f'Router listening on port {PORT}, backends: {[x["name"] for x in items]}', flush=True)
    server.serve_forever()
