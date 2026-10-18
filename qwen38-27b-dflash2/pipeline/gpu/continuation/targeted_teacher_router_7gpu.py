"""Simple HTTP router: load-balances across N TP1 teacher replicas on one stable port.

Adapted from the 4B-line TeacherRouter (dynamic_teacher_pool.py) but simplified:
no dynamic membership, no health probes — just round-robin least-active dispatch
to a fixed set of backends. Runs on port 8250 (same port the trainer expects)."""
import hashlib
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

PORT = 8250
BACKENDS = {0: 8260, 1: 8261, 2: 8262}
LOG = Path(os.environ['TARGETED_RUNTIME_DIR'])/'router_routing.jsonl'
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
        gpu, port, seq = router.pick()
        code = 502
        try:
            req = urllib.request.Request(
                f'http://127.0.0.1:{port}' + self.path, data=payload,
                method=self.command, headers={'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(req, timeout=610)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                content = response.read()
                code = response.status
                self.send_response(code)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception as exc:
            try:
                self.send_error(502, str(exc))
            except OSError:
                pass
        finally:
            router.done(gpu, seq, code)


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


if __name__ == '__main__':
    server = Server(('127.0.0.1', PORT), Handler)
    print(f'Router listening on port {PORT}, backends: {BACKENDS}', flush=True)
    server.serve_forever()
