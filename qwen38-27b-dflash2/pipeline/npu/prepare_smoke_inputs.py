"""Copy the first 128 raw conversation rows to a fresh smoke input directory."""
import argparse
import json
from pathlib import Path


def select(inputs, output, count=128):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for path in inputs:
        with path.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not (isinstance(row.get('messages'), list) or
                                                      isinstance(row.get('conversations'), list)):
                    raise ValueError(f'Expected a raw conversation in {path}')
                rows.append(row)
                if len(rows) == count:
                    break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f'Need {count} raw conversation rows; found {len(rows)}')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'prompts128.jsonl').open('x', encoding='utf-8') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
    return output / 'prompts128.jsonl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(select(args.input, args.output))
