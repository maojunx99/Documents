"""Read-only token/mask/ID/row-count audit; keep length-truncated answers."""
from pathlib import Path
import argparse
import collections
import json


def verify(folder, expected_rows=None, expected_sampling=None, expected_primary_ids=None):
    files = sorted(Path(folder).glob('*_out.jsonl'))
    if not files:
        raise ValueError('No regeneration output shards found')
    seen = set()
    primary = set()
    finish = collections.Counter()
    rows = 0
    for path in files:
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                ids, mask = row['input_ids'], row['loss_mask']
                if not ids or len(ids) != len(mask) or not any(mask) or any(x not in (0, 1, False, True) for x in mask):
                    raise ValueError(f'Invalid input or supervision mask: {path}')
                key = row['id']
                if key in seen:
                    raise ValueError('Duplicate generated record ID: ' + str(key))
                seen.add(key)
                primary.add(str(row.get('primary_id', key)))
                meta = row['metadata']
                if expected_sampling and any(meta['sampling_params'].get(k) != v for k, v in expected_sampling.items()):
                    raise ValueError('Recorded sampling parameters differ')
                finish[str(meta.get('finish_reason'))] += 1
                rows += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f'Generated {rows} rows, expected {expected_rows}')
    if expected_primary_ids is not None and primary != set(expected_primary_ids):
        raise ValueError(f'Conversation coverage differs: missing={len(set(expected_primary_ids)-primary)}, extra={len(primary-set(expected_primary_ids))}')
    return {'passed': True, 'rows': rows, 'conversations': len(primary), 'shards': len(files), 'finish_reasons': dict(finish),
            'policy': 'Keep finish_reason=length records; training preparation applies its separate sequence cap.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--expected-rows', type=int)
    a = p.parse_args()
    print(json.dumps(verify(a.shards, a.expected_rows), indent=2))
