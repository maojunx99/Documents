"""Isolated selector calibration experiment; no change to candidate policy."""
import torch
from functools import partial


def overlap_elements(logits, teacher_log_probs):
    # Full-vocabulary normalized teacher mass restricted to the same K candidates.
    # q is zero outside K; 1 - overlap equals full-vocabulary TV, including p outside K.
    p = teacher_log_probs.detach().float().exp()
    q = logits.float().softmax(-1)
    return 1.0 - torch.minimum(p, q).sum(-1)


def overlap_selector_loss(candidate_logits, teacher_logits, loss_mask, block_size,
                       *, gamma, per_position_loss_weight, dpace_alpha,
                       sample_from_anchor=False):
    from speculators.losses import dflash_loss_decay, loss_function
    if per_position_loss_weight != 'fixed-exp-decay':
        raise ValueError('This experiment is locked to fixed-exp-decay')
    pos = (torch.arange(candidate_logits.shape[1], device=candidate_logits.device)
           % block_size).unsqueeze(0)
    return loss_function(candidate_logits, teacher_logits.detach(), loss_mask, pos,
                         loss_fn=overlap_elements,
                         decay_fn=partial(dflash_loss_decay, gamma=gamma,
                                          sample_from_anchor=sample_from_anchor))


@torch.no_grad()
def native_metrics(selector, unary, hidden, predecessor, native_ids, targets,
                   mask, block_size):
    # Scores the *unmodified* top-K used by inference, even if training injected
    # the teacher argmax. This is teacher-forced, unfiltered T=1 diagnostics,
    # not serving acceptance length and not a generated rollout estimate.
    scores = selector.score_candidates(unary.detach(), hidden.detach(),
                                        predecessor.reshape(1, -1), native_ids)
    logq = scores.float().log_softmax(-1)
    q = logq.exp()
    teacher_k = targets.detach().gather(-1, native_ids).float()
    logpk = teacher_k.log_softmax(-1)
    pk = logpk.exp()
    mass = (teacher_k.logsumexp(-1) - targets.detach().float().logsumexp(-1)).exp()
    overlap = torch.minimum(q, pk * mass.unsqueeze(-1)).sum(-1)
    entropy = -(q * logq).sum(-1)
    kl = (pk * (logpk - logq)).sum(-1)
    tv = 0.5 * (pk - q).abs().sum(-1)
    vf = mask.float()
    out = {}
    values = {'native_candidate_mass': mass, 'native_selector_entropy': entropy,
              'native_conditional_kl': kl, 'native_conditional_tv': tv,
              'native_tf_unfiltered_overlap': overlap}
    for name, value in values.items():
        out[name + '_sum'] = (value * vf).sum()
        out[name + '_total'] = vf.sum()
    for pos in range(1, block_size):
        pm = vf[:, pos::block_size]
        for name in ['native_candidate_mass', 'native_tf_unfiltered_overlap']:
            out[f'{name}_pos{pos}_sum'] = (values[name][:, pos::block_size] * pm).sum()
            out[f'{name}_pos{pos}_total'] = pm.sum()
    return out
