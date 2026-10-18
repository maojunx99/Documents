"""Fail instead of changing native loss support when a pack boundary is crossed."""
import torch


@torch.compiler.disable
def assert_packed_supervision(input_ids, loss_mask, document_ids, block_indices,
                              aligned_loss_mask, block_size=8):
    if input_ids.ndim!=2 or input_ids.shape[0]!=1:
        raise ValueError('Only the sealed single packed sequence is supported')
    if loss_mask.shape!=input_ids.shape or document_ids.shape!=input_ids.shape:
        raise ValueError('Packed mask/document shape mismatch')
    if block_indices.ndim!=1 or not len(block_indices) or len(block_indices)%block_size:
        raise ValueError('Invalid anchored block indices')
    if aligned_loss_mask.shape!=(1,len(block_indices)):
        raise ValueError('Aligned supervision shape mismatch')
    if not torch.all((aligned_loss_mask==0)|(aligned_loss_mask==1)).item():
        raise ValueError('Nonbinary aligned loss support')
    n=input_ids.shape[1]
    if not torch.all((block_indices>=0)&(block_indices<n)).item():
        raise ValueError('Anchor block outside packed input')
    idx=block_indices.reshape(-1,block_size)
    if not torch.equal(idx,idx[:,:1]+torch.arange(block_size,device=idx.device)):
        raise ValueError('Anchor blocks are not consecutive recorded token positions')
    active=aligned_loss_mask[0].bool().reshape_as(idx)
    if active[:,0].any().item():raise ValueError('Anchor itself cannot receive training loss')
    docs=document_ids[0]
    anchor_docs=docs[idx[:,:1]]
    target_docs=docs[idx]
    predecessor_docs=docs[(idx-1).clamp_min(0)]
    same=(target_docs==anchor_docs)&(predecessor_docs==anchor_docs)&(anchor_docs>=0)&(idx>0)
    if (active&~same).any().item():
        raise ValueError('Positive loss crosses a document/padding boundary or lacks predecessor')
    if (active&~loss_mask[0,idx].bool()).any().item():
        raise ValueError('Aligned loss activates an unsupervised recorded token')
    return dict(valid_positions=int(active.sum()),
                positive_cross_document_positions=0,
                target_shift='recorded input_ids[i], teacher last_hidden[i-1]',
                input_tokens=int((document_ids>=0).sum()))
