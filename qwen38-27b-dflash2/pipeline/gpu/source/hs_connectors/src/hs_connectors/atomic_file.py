"""Publish completed feature files on a shared mount; no distributed flock."""
import json
import os
from pathlib import Path
import time
import uuid


def publish(filename, save):
    final = Path(filename)
    temporary = final.with_name('.'+final.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        save(str(temporary))
        with temporary.open('r+b') as f:
            os.fsync(f.fileno())
        os.replace(temporary, final)
        if os.name == 'posix':
            fd = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException as exc:
        Path(str(final)+'.failed').write_text(json.dumps({'error':repr(exc)}))
        raise


def wait(filename, timeout=900):
    path = Path(filename)
    deadline = time.monotonic()+timeout
    while not path.exists():
        error = Path(str(path)+'.failed')
        if error.exists():
            raise RuntimeError('Shared-file teacher writer failed: '+error.read_text())
        if time.monotonic()>deadline:
            raise TimeoutError('Shared-file feature did not arrive: '+str(path))
        time.sleep(.02)
    return path
