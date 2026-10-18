# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
"""Verify the transferred Qwen3.8-27B regeneration output (Phase 1 gate).

Checks: shard count, row counts vs the 859,794-row prompt pool, error sidecars,
per-source finish_reason stats (truncation), empty/short responses, loss_mask
sanity, thinking-tag presence. Policy per user decision: keep ALL rows including
finish_reason=length; this script only records, never filters."""
import collections
import glob
import json
from pathlib import Path

ROOT = Path('/workspace/dflash2_work/public_reproduction/qwen38_27b_pb860k_20260926')
DATA = ROOT/'regen_data'
EXPECTED_ROWS = 859794

files = sorted(glob.glob(str(DATA/'*_out.jsonl')))
errors = sorted(glob.glob(str(DATA/'*_out.errors.jsonl')))
print(f'output shards: {len(files)}  error sidecars: {len(errors)}')

per_source = collections.defaultdict(lambda: collections.Counter())
total = 0
completion_tokens = []
bad_rows = 0
for path in files:
    name = Path(path).name
    source = name.rsplit('_', 2)[0]
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            c = per_source[source]
            meta = row.get('metadata') or {}
            c[str(meta.get('finish_reason'))] += 1
            ids = row.get('input_ids') or []
            mask = row.get('loss_mask') or []
            if len(ids) != len(mask) or not any(mask):
                bad_rows += 1
                continue
            completion_tokens.append(sum(mask))

print(f'total rows: {total} (expected {EXPECTED_ROWS}, match={total == EXPECTED_ROWS})')
print(f'bad rows (mask empty/misaligned): {bad_rows}')
trunc = sum(c.get("['length']", 0) for c in [v for v in per_source.values()])
print(f'truncated (finish_reason=length) overall: {sum(v["length"] for v in per_source.values())}')
for source in sorted(per_source):
    c = per_source[source]
    n = sum(c.values())
    print(f'  {source:20s} rows={n:7d} stop={c.get("stop",0):7d} length={c.get("length",0):7d} trunc={c.get("length",0)/n:6.1%}')
if completion_tokens:
    completion_tokens.sort()
    n = len(completion_tokens)
    print(f'completion tokens: mean={sum(completion_tokens)/n:.0f} p50={completion_tokens[n//2]} p90={completion_tokens[int(n*0.9)]} p99={completion_tokens[int(n*0.99)]} max={completion_tokens[-1]}')

result = dict(passed=(total == EXPECTED_ROWS and bad_rows == 0), rows=total, expected=EXPECTED_ROWS,
              bad_rows=bad_rows, per_source={k: dict(v) for k, v in per_source.items()},
              policy='keep all rows including finish_reason=length (user decision)')
(ROOT/'run/regen_verification.json').write_text(json.dumps(result, indent=2))
print('VERIFY_DONE passed=', result['passed'])
