"""Selector probability matching, independent of accelerator-specific kernels."""

import torch


@torch.no_grad()
def teacher_log_normalizer(logits: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
    """Normalize over the full vocabulary with bounded FP32 temporary storage.

    Only a row chunk is promoted to FP32, rather than the complete [N, V]
    teacher tensor. The result is detached and has shape logits.shape[:-1].
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    rows = logits.reshape(-1, logits.shape[-1])
    if rows.shape[0] == 0:
        return logits.new_empty(logits.shape[:-1], dtype=torch.float32)
    return torch.cat(
        [
            torch.logsumexp(rows[start : start + chunk_rows].float(), dim=-1)
            for start in range(0, rows.shape[0], chunk_rows)
        ]
    ).reshape(logits.shape[:-1])


def full_teacher_tv_elements(
    candidate_logits: torch.Tensor, teacher_candidate_log_probs: torch.Tensor
) -> torch.Tensor:
    """Full-vocabulary TV with draft probability zero outside the K candidates.

    Teacher log probabilities must use the *full-vocabulary*, T=1 normalizer.
    They are not renormalized within K. Only the teacher branch is detached;
    gradients still flow through candidate scores into the draft backbone.
    """
    p = teacher_candidate_log_probs.detach().float().exp()
    q = candidate_logits.float().softmax(dim=-1)
    return 1.0 - torch.minimum(p, q).sum(dim=-1)
