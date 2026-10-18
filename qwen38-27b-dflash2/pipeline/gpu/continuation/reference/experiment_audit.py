"""Small per-rank input/anchor audit, outside compiled graphs."""
import hashlib
import json
import os
from pathlib import Path
import torch

PHASE = 'unset'
STEP = 0
VAL_INDEX = 0


def tensor_hash(x):
    x = x.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.compiler.disable
def audit_batch(model, input_ids, loss_mask, block_indices, document_ids):
    global VAL_INDEX
    if PHASE == 'unset':
        return
    rank = int(os.environ.get('RANK', '0'))
    d = Path(os.environ['EXPERIMENT_ARM_DIR']) / 'audit'
    d.mkdir(exist_ok=True, parents=True)
    rec = {'phase': PHASE, 'step': STEP if PHASE == 'train' else VAL_INDEX,
           'input_sha256': tensor_hash(input_ids), 'mask_sha256': tensor_hash(loss_mask),
           'anchors_sha256': tensor_hash(block_indices),
           'documents_sha256': tensor_hash(document_ids),
           'input_tokens': int((document_ids >= 0).sum()),
           'supervised_tokens': int(loss_mask.sum())}
    with (d / f'inputs_rank{rank}.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    if PHASE != 'train':
        VAL_INDEX += 1
