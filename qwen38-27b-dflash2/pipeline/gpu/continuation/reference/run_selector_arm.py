"""Native trainer with fixed paired baseline/final validation and step audit."""
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import experiment_audit as audit
from speculators.train.trainer import Trainer
from speculators.train.cli import main
from speculators.train.config import TrainConfig


ROOT = Path(os.environ['EXPERIMENT_ARM_DIR'])
RANK = int(os.environ.get('RANK', '0'))
(ROOT / 'audit').mkdir(exist_ok=True, parents=True)


def write(name, data):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(p)


old_run = Trainer.run_training
old_val = Trainer.val_epoch
old_step = Trainer._optimizers_step


def fixed_val(self, epoch):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    phase = audit.PHASE
    audit.PHASE = 'baseline' if self.global_step == 0 else 'final'
    audit.VAL_INDEX = 0
    random.seed(20261008)
    np.random.seed(20261008)
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    try:
        metrics = old_val(self, 0)
        write(f'audit/{audit.PHASE}_rank{RANK}.json', {
            'global_step': self.global_step, 'batches': audit.VAL_INDEX,
            'metrics': metrics, 'time': time.time()})
        return metrics
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        audit.PHASE = phase


def step(self):
    old_step(self)
    completed = self.global_step + 1
    audit.STEP = completed
    if completed % 10 == 0 or completed == 1:
        write(f'audit/progress_rank{RANK}.json', {
            'completed_optimizer_steps': completed, 'time': time.time(),
            'lr': [o.param_groups[0]['lr'] for o in self.optimizers]})


def run(self):
    assert self.global_step == 0 and self.current_epoch == 0
    assert self.config.max_steps == 500 and self.config.num_epochs == 1
    assert self.config.scheduler_total_steps == 500
    assert all(len(o.state) == 0 for o in self.optimizers)
    model = self.model.module if hasattr(self.model, 'module') else self.model
    initial = {n: audit.tensor_hash(p) for n, p in model.named_parameters()
               if p.requires_grad}
    write(f'audit/initial_rank{RANK}.json', {
        'parameter_sha256': initial, 'num_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'optimizer_states': [len(o.state) for o in self.optimizers],
        'train_batches_full_epoch': len(self.train_loader),
        'val_batches': len(self.val_loader), 'workers': self.train_loader.num_workers,
        'pid': os.getpid(), 'mode': os.environ['SELECTOR_OBJECTIVE'],
        'config': self.config._asdict()})
    fixed_val(self, 0)
    if dist.is_initialized():
        dist.barrier()
    audit.PHASE = 'train'
    audit.STEP = 0
    old_run(self)
    assert self.global_step == 500
    write(f'audit/completed_rank{RANK}.json', {'optimizer_steps': self.global_step,
                                               'time': time.time()})


Trainer.run_training = run
Trainer.val_epoch = fixed_val
Trainer._optimizers_step = step

if __name__ == '__main__':
    try:
        main(TrainConfig.resolve())
    except BaseException:
        write(f'audit/failure_rank{RANK}.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise
