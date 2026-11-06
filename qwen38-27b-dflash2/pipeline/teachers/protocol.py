"""Shared-file metadata checks; no authentication or HTTP gateway."""
import array
import hashlib
import json
import os
from pathlib import Path
import struct
from urllib.parse import urlsplit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_peers(path):
    if not path:
        return []
    raw = json.loads(Path(path).read_text())
    if set(raw) != {'teachers'} or not raw['teachers']:
        raise ValueError('Expected a nonempty teachers list')
    result = []
    for item in raw['teachers']:
        if set(item) != {'name', 'url', 'shared_root'}:
            raise ValueError('Native teacher needs only name/url/shared_root; no API key')
        url = urlsplit(item['url'])
        if (url.scheme != 'http' or not url.hostname or url.username or url.password
                or url.path not in ['', '/'] or url.query or url.fragment
                or not Path(item['shared_root']).is_absolute()):
            raise ValueError('Use a native teacher base URL and an absolute shared cache path')
        if not item['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in item['name']):
            raise ValueError('Invalid teacher name')
        result.append(dict(item, url=item['url'].rstrip('/')))
    for key in ['name','url']:
        if len({x[key] for x in result}) != len(result):
            raise ValueError('Duplicate teacher '+key)
    return result


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
