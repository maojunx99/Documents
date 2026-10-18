import json,os,pathlib
import torch
import torch_npu
from transformers import PreTrainedModel,Qwen3Config
from safetensors.torch import load_file
from speculators.train.checkpointer import SingleGPUCheckpointer
import cpu_checkpoint

class Tiny(PreTrainedModel):
    config_class=Qwen3Config
    _keys_to_ignore_on_save=['frozen.weight']
    def __init__(self,config):
        super().__init__(config)
        self.body=torch.nn.Linear(4,4)
        self.frozen=torch.nn.Embedding(8,4);self.frozen.weight.requires_grad_(False)

torch.manual_seed(42);torch.npu.set_compile_mode(jit_compile=False)
cpu_checkpoint.install()
root=pathlib.Path(os.environ['PREFLIGHT_ROOT']);root.mkdir(exist_ok=False)
model=Tiny(Qwen3Config()).to('npu:0');opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
model.body(torch.randn(2,4,device='npu:0')).square().mean().backward();opt.step()
before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
c=SingleGPUCheckpointer(str(root));c.save_checkpoint(model,[opt],0)
state=load_file(str(root/'0/model.safetensors'))
assert set(state)=={'body.weight','body.bias'}
assert all(v.dtype==torch.bfloat16 and torch.equal(v,before[k].to(torch.bfloat16)) for k,v in state.items())
assert all(torch.equal(p.detach().cpu(),before[n]) for n,p in model.named_parameters())
saved=torch.load(root/'0/optimizer_state_dict.pt',map_location='cpu',weights_only=False)
assert all(v.device.type=='cpu' for s in saved['state'].values() for v in s.values() if isinstance(v,torch.Tensor))
reloaded=Tiny(Qwen3Config());reloaded.load_state_dict(state,strict=False)
assert all(torch.equal(dict(reloaded.named_parameters())[n],v.float()) for n,v in state.items())
result={'passed':True,'actual_npu_model_and_optimizer_saved':True,'host_dtype_conversion':True,'live_fp32_parameters_unchanged':True,'bf16_serialization_and_cpu_reload_exact':True}
pathlib.Path(os.environ['AUDIT_OUT']).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
