"""CPU-only transport checks. Never rely on a cross-host flock."""
import array
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_key(path):
    p = Path(path)
    if os.name == 'posix' and p.stat().st_mode & 0o077:
        raise ValueError('Teacher API key file must have mode 0600')
    key = p.read_text().strip()
    if len(key) < 32 or any(c.isspace() for c in key):
        raise ValueError('Use a private random teacher API key of at least 32 characters')
    return key


def load_peers(path, require_keys=True):
    if not path:
        return []
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    if set(raw) != {'teachers'} or not isinstance(raw['teachers'], list) or not raw['teachers']:
        raise ValueError('remote-teachers must contain a nonempty teachers list')
    peers, names, urls, roots = [], set(), set(), set()
    for item in raw['teachers']:
        if set(item) != {'name', 'url', 'api_key_file', 'shared_root'}:
            raise ValueError('Each remote teacher needs name/url/api_key_file/shared_root')
        url = urlsplit(item['url'])
        try:
            port = url.port
        except ValueError:
            raise ValueError('Invalid remote teacher URL port') from None
        name = item['name']
        if (not isinstance(name, str) or not name or not all(c.isalnum() or c in '-_' for c in name)
                or url.scheme not in ['http', 'https'] or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path not in ['', '/']):
            raise ValueError('Invalid teacher name or base URL; credentials belong in api_key_file')
        if port is not None and not 1 <= port <= 65535:
            raise ValueError('Invalid remote teacher URL port')
        peer = dict(item, url=item['url'].rstrip('/'))
        if not Path(peer['shared_root']).is_absolute():
            raise ValueError('shared_root must be an absolute, identically mounted path')
        if name in names or peer['url'] in urls or peer['shared_root'] in roots:
            raise ValueError('Duplicate teacher name, URL, or shared root')
        names.add(name); urls.add(peer['url']); roots.add(peer['shared_root'])
        key_path = Path(peer['api_key_file'])
        peer['api_key_file'] = str((config_path.parent / key_path).resolve())
        if require_keys:
            read_key(peer['api_key_file'])
        peers.append(peer)
    return peers


def model_manifest(model):
    """Hash actual weights and tokenizer/config bytes, not host-specific mtimes."""
    root = Path(model).resolve()
    paths = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in ['.safetensors', '.json', '.model', '.txt', '.jinja'] or p.name == 'merges.txt'))
    if not any(p.suffix == '.safetensors' for p in paths):
        raise ValueError('A complete safetensors target is required')
    return {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths}


def source_manifest(backend):
    source = ROOT / backend / 'source'
    paths = sorted(p for p in source.rglob('*.py') if '__pycache__' not in p.parts)
    result = {str(p.relative_to(source)).replace('\\', '/'): digest(p) for p in paths}
    for p in sorted((ROOT/'teachers').glob('*.py')) + [ROOT/'common/managed.py']:
        result['__pipeline__/'+str(p.relative_to(ROOT)).replace('\\','/')] = digest(p)
    return result


def runtime_manifest(backend):
    import importlib.metadata
    names = ['torch', 'transformers', 'vllm'] + (['torch-npu', 'vllm-ascend'] if backend == 'npu' else [])
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return versions


def tensor_file(path, layers, width, expected_tokens=None, finite=False):
    """Validate complete safetensors byte ranges and exact input IDs without torch."""
    path = Path(path)
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete hidden-state file prefix')
        size = struct.unpack('<Q', prefix)[0]
        if not 8 <= size <= 1024**2:
            raise ValueError('Invalid hidden-state header length')
        raw = f.read(size)
        header = json.loads(raw)
        if set(header) - {'__metadata__'} != {'hidden_states', 'token_ids'}:
            raise ValueError('Unexpected hidden-state tensor keys')
        hidden, tokens = header['hidden_states'], header['token_ids']
        shape = hidden['shape']
        if (hidden['dtype'] != 'BF16' or len(shape) != 3 or shape[0] < 1
                or shape[1:] != [layers, width] or tokens['dtype'] != 'I64'
                or tokens['shape'] != [shape[0]]):
            raise ValueError('Hidden-state dtype/shape differs from the target contract')
        ranges = []
        for tensor, itemsize in [(hidden, 2), (tokens, 8)]:
            count = 1
            for dim in tensor['shape']:
                if not isinstance(dim, int) or dim < 1:
                    raise ValueError('Invalid tensor dimension')
                count *= dim
            lo, hi = tensor['data_offsets']
            if lo < 0 or hi - lo != count * itemsize:
                raise ValueError('Invalid tensor byte range')
            ranges.append((lo, hi))
        ranges.sort()
        if ranges[0][0] != 0 or ranges[0][1] != ranges[1][0] or path.stat().st_size != 8 + size + ranges[-1][1]:
            raise ValueError('Incomplete or overlapping hidden-state payload')
        if expected_tokens is not None:
            if len(expected_tokens) != shape[0]:
                raise ValueError('Hidden-state input length mismatch')
            f.seek(8 + size + tokens['data_offsets'][0])
            ids = array.array('q'); ids.frombytes(f.read(shape[0] * 8))
            if __import__('sys').byteorder != 'little':
                ids.byteswap()
            if ids.tolist() != expected_tokens:
                raise ValueError('Hidden-state token IDs differ from the request')
        if finite:
            f.seek(8 + size + hidden['data_offsets'][0])
            remaining = hidden['data_offsets'][1] - hidden['data_offsets'][0]
            while remaining:
                data = f.read(min(1024**2, remaining))
                if not data:
                    raise ValueError('Hidden-state payload was truncated during the finite probe')
                values = array.array('H'); values.frombytes(data)
                if __import__('sys').byteorder != 'little': values.byteswap()
                if any((value & 0x7f80) == 0x7f80 for value in values):
                    raise ValueError('Nonfinite teacher hidden states')
                remaining -= len(data)
    return {'bytes': path.stat().st_size, 'tokens': shape[0], 'layers': layers, 'width': width}


def wait_local_writer(path, deadline):
    lock = Path(str(path) + '.lock')
    while not path.exists() and not lock.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Hidden-state writer did not start')
        time.sleep(.01)
    if lock.exists():
        import fcntl  # The publisher and writer run on the SAME Linux host.
        if lock.is_symlink():
            raise ValueError('Writer lock must not be a symlink')
        fd = os.open(lock, os.O_RDWR)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Hidden-state writer did not complete')
                    time.sleep(.01)
        finally:
            os.close(fd)


def publish_complete(handle, staging, ready, layers, width, tokens, timeout=180, waiter=wait_local_writer):
    source = Path(handle)
    if source.is_symlink() or source.suffix != '.safetensors' or source.parent.resolve() != Path(staging).resolve():
        raise ValueError('Teacher returned a handle outside its owned staging directory')
    waiter(source, time.monotonic() + timeout)
    with source.open('r+b') as f:
        os.fsync(f.fileno())
    receipt = tensor_file(source, layers, width, tokens)
    ready = Path(ready)
    destination = ready / (uuid.uuid4().hex + '.safetensors')
    # Same-filesystem rename: a trainer never sees an in-progress file/lock.
    os.replace(source, destination)
    if os.name == 'posix':
        fd = os.open(ready, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    lock = Path(str(source) + '.lock')
    if lock.exists():
        lock.unlink()
    return str(destination), receipt
