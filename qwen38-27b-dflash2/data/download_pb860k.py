#!/usr/bin/env python3
"""Download, verify and extract the original Qwen3.8-27B PB860K regenerated data.

Uses the Python standard library only. Existing verified files are reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

BASE = 'https://github.com/maojunx99/Documents/releases/download/pb860k-regenerated-20261022/'
MANIFEST_SHA256 = '1f0dd2e29f367040af03aadc8e4f0d03cec856dd059e8e09d04337b7a95c2f4e'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def download(part, directory):
    p = directory / part['name']
    if p.exists():
        if p.stat().st_size != part['bytes'] or digest(p) != part['sha256']:
            raise ValueError(f'Existing file fails checksum: {p}. Move it aside before retrying.')
        print(f'Already verified: {p.name}', flush=True)
        return p
    temp = p.with_suffix(p.suffix + '.partial')
    for attempt in range(3):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            if offset > part['bytes']:
                raise ValueError(f'Partial download exceeds expected size: {temp}')
            if offset < part['bytes']:
                headers = {'User-Agent': 'pb860k-dataset-downloader'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                req = urllib.request.Request(BASE + p.name, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as response:
                    if offset and response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected Content-Range')
                        mode = 'ab'
                    else:
                        mode = 'wb'
                    with temp.open(mode) as out:
                        shutil.copyfileobj(response, out, 1024**2)
            if temp.stat().st_size != part['bytes'] or digest(temp) != part['sha256']:
                raise ValueError(f'Download checksum mismatch: {temp}')
            os.replace(temp, p)
            print(f'Downloaded and verified: {p.name}', flush=True)
            return p
        except (OSError, TimeoutError) as exc:
            if attempt == 2:
                raise
            print(f'Retrying {p.name}: {exc}', flush=True)
            time.sleep(5)


def extract(part, archive, output):
    expected = {m['path']: m for m in part['members']}
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected archive entry: {member.name}')
            seen.add(member.name)
            item = expected[member.name]
            target = (output / member.name).resolve()
            target.relative_to(output.resolve())
            if member.size != item['bytes']:
                raise ValueError(f'Unexpected entry size: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size == item['bytes'] and digest(target) == item['sha256']:
                    continue
                raise ValueError(f'Existing extracted file fails checksum: {target}')
            temporary = target.with_suffix(target.suffix + '.partial')
            h = hashlib.sha256()
            with tar.extractfile(member) as source, temporary.open('wb') as dest:
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    h.update(block)
                    dest.write(block)
            if temporary.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
                raise ValueError(f'Extracted checksum mismatch: {member.name}')
            os.replace(temporary, target)
    if seen != set(expected):
        raise ValueError('Archive is missing expected members')
    print(f'Extracted and verified: {archive.name}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('pb860k'))
    ap.add_argument('--workers', type=int, default=2, choices=range(1, 9), metavar='1..8')
    ap.add_argument('--download-only', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'DATASET_MANIFEST.json'
    if manifest_path.exists():
        raw = manifest_path.read_bytes()
    else:
        with urllib.request.urlopen(BASE + 'DATASET_MANIFEST.json', timeout=90) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Dataset manifest checksum mismatch')
    manifest = json.loads(raw)
    if manifest['rows'] != 859794 or not all(manifest['verification'].values()):
        raise ValueError('Unexpected dataset manifest')
    manifest_path.write_bytes(raw)
    archives = args.output / 'archives'
    archives.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(lambda p: download(p, archives), manifest['parts']))
    if not args.download_only:
        for part, archive in zip(manifest['parts'], paths):
            extract(part, archive, args.output)
    print(f"Complete: {manifest['rows']:,} records; {len(paths)} verified archives.")
    if not args.download_only:
        print(f"Training source: {(args.output / 'regen_data').resolve()}")


if __name__ == '__main__':
    main()
