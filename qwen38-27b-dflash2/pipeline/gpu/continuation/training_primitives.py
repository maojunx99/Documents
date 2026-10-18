"""Auditable objective/optimizer building blocks, not a runnable trainer.

Native normalization retained per PLAN_IMPLEMENTATION_ADDENDUM_v1.json.
Integration, DDP/no_sync, shared hidden and GPU smoke remain separate gates.
"""
import torch
import torch.distributed as dist

EPS=1e-5


def position_weights(mask,block_size=8,gamma=4.0):
    pos=torch.arange(mask.shape[-1],device=mask.device)%block_size
    decay=torch.exp(-(pos-1).clamp_min(0).float()/gamma)*(pos!=0)
    return mask.float()*decay


def recorded_labels(input_ids,block_indices,*,sample_from_anchor=False):
    if sample_from_anchor:
        raise ValueError('Locked DFlash2 experiment uses sample_from_anchor=False')
    # Backbone aligns verifier hidden at index-1 to predict input_ids[index].
    # Slot0 is the visible anchor and is excluded by aligned_loss_mask.
    return input_ids[:,block_indices]


def overlap_numerator(candidate_logits,teacher_logits,candidate_ids,mask,*,block_size=8,gamma=4.0):
    logpk=teacher_logits.detach().gather(-1,candidate_ids).float()-teacher_logits.detach().float().logsumexp(-1,keepdim=True)
    elements=1-torch.minimum(logpk.exp(),candidate_logits.float().softmax(-1)).sum(-1)
    # Preserve native operation order for exact single-pack parity.
    numerator=(elements*mask.float()*(position_weights(torch.ones_like(mask),block_size,gamma))).sum()
    return numerator,mask.float().sum()


def sampled_ce_numerator(candidate_logits,candidate_ids,actual_ids,mask,*,block_size=8,gamma=4.0):
    matches=candidate_ids.eq(actual_ids.unsqueeze(-1))
    covered=matches.any(-1)
    slots=matches.long().argmax(-1)
    ce=-candidate_logits.float().log_softmax(-1).gather(-1,slots.unsqueeze(-1)).squeeze(-1)
    elements=torch.where(covered,ce,torch.zeros_like(ce))
    numerator=(elements*mask.float()*position_weights(torch.ones_like(mask),block_size,gamma)).sum()
    denominator=mask.float().sum()  # Includes valid, absent-support labels.
    return numerator,denominator,covered


class FP32MasterAdamW:
    """BF16 model, FP32 master/Adam/accumulation; explicit numerator gradients.

    Caller wraps *every* microbatch forward/backward in DDP.no_sync(), uses
    unnormalized loss numerator, then calls capture_microbatch(valid_count).
    Exactly four calls precede step(). Buffers are summed across ranks once,
    divided by total valid count, clipped once, updated once and copied back.
    Nothing in this class silently decides DDP synchronization or tail policy.
    """
    def __init__(self,named_parameters,lr=1e-4,microbatches=4):
        self.model=[(n,p) for n,p in named_parameters if p.requires_grad]
        if not self.model:raise ValueError('No trainable parameters')
        self.master=[torch.nn.Parameter(p.detach().float().clone()) for _,p in self.model]
        self.optimizer=torch.optim.AdamW(self.master,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        self.expected_microbatches=microbatches
        self.microbatches=0;self.completed_steps=0
        self.denominator=torch.zeros((),device=self.master[0].device,dtype=torch.float64)
        self.buffers=[torch.zeros_like(p) for p in self.master]

    @torch.no_grad()
    def capture_microbatch(self,valid_count):
        if self.microbatches>=self.expected_microbatches:raise RuntimeError('Accumulation overflow')
        self.denominator.add_(torch.as_tensor(valid_count,device=self.denominator.device,dtype=torch.float64))
        for (n,p),b in zip(self.model,self.buffers,strict=True):
            if p.grad is None:raise RuntimeError('Missing gradient: '+n)
            if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient: '+n)
            b.add_(p.grad.float());p.grad=None
        self.microbatches+=1

    @torch.no_grad()
    def step(self,*,distributed=False):
        if self.microbatches!=self.expected_microbatches:raise RuntimeError('Incomplete accumulation window')
        if distributed:
            if not dist.is_initialized():raise RuntimeError('Missing DDP process group')
            dist.all_reduce(self.denominator,op=dist.ReduceOp.SUM)
            for b in self.buffers:dist.all_reduce(b,op=dist.ReduceOp.SUM)
        if not torch.isfinite(self.denominator) or self.denominator<=0:raise RuntimeError('Invalid denominator')
        denominator=self.denominator.float()+EPS
        for master,b in zip(self.master,self.buffers,strict=True):master.grad=b/denominator
        grad_norm=torch.nn.utils.clip_grad_norm_(self.master,1.0,error_if_nonfinite=True)
        self.optimizer.step();self.completed_steps+=1
        for (_,p),master in zip(self.model,self.master,strict=True):p.copy_(master)
        self.optimizer.zero_grad(set_to_none=True)
        for b in self.buffers:b.zero_()
        self.denominator.zero_();self.microbatches=0
        return float(grad_norm)

    def state_dict(self):
        if self.microbatches:raise RuntimeError('Checkpoint only at completed optimizer boundaries')
        return {'names':[n for n,_ in self.model], 'master':[p.detach().cpu() for p in self.master],
                'optimizer':self.optimizer.state_dict(),'completed_steps':self.completed_steps,
                'expected_microbatches':self.expected_microbatches}

    @torch.no_grad()
    def load_state_dict(self,state):
        if self.microbatches:raise RuntimeError('Cannot restore over partial gradients')
        assert state['names']==[n for n,_ in self.model]
        assert state['expected_microbatches']==self.expected_microbatches
        for (_,p),master,saved in zip(self.model,self.master,state['master'],strict=True):
            assert saved.dtype==torch.float32 and saved.shape==master.shape
            master.copy_(saved);p.copy_(master)
        self.optimizer.load_state_dict(state['optimizer']);self.completed_steps=state['completed_steps']
        for state in self.optimizer.state.values():
            for k in ('exp_avg','exp_avg_sq'):assert state[k].dtype==torch.float32
