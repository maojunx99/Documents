"""Native-compatible checkpoint format with dtype conversion on host memory."""
import torch
import torch.distributed as dist
from torch.utils import _pytree
from speculators.train.checkpointer import SingleGPUCheckpointer, patch_config_dtype

def host_state(tree,dtype):
    def convert(value):
        if not isinstance(value,torch.Tensor):return value
        value=value.detach().cpu()
        return value.to(dtype) if value.is_floating_point() else value
    return _pytree.tree_map(convert,tree)

def cpu_save_checkpoint(self,model,optimizer,epoch,float_dtype=torch.bfloat16):
    if dist.is_initialized() and dist.get_rank()!=0:return
    raw=model.module if hasattr(model,'module') else model
    state=raw.state_dict()
    for name in getattr(raw,'_keys_to_ignore_on_save',None) or []:state.pop(name,None)
    model_state=host_state(state,float_dtype)
    raw.save_pretrained(self.path/str(epoch),state_dict=model_state)
    patch_config_dtype(self.path/str(epoch)/'config.json',float_dtype)
    del model_state,state
    optimizers=list(optimizer) if isinstance(optimizer,(list,tuple)) else [optimizer]
    states=[host_state(opt.state_dict(),float_dtype) for opt in optimizers]
    torch.save(states[0] if len(states)==1 else states,self.optimizer_path(epoch))
    self._copy_train_command(epoch)

def install():SingleGPUCheckpointer.save_checkpoint=cpu_save_checkpoint
