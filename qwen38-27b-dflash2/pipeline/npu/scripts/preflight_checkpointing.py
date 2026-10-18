"""Check decoder checkpointing with identical inputs, weights and gradients."""
import copy, functools, json, os, pathlib
import torch
import torch_npu
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer

def check(device,dtype):
    torch.manual_seed(42)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        layer_types=['full_attention']*2,attention_dropout=0.,max_position_embeddings=128)
    cfg._attn_implementation='sdpa'
    layers=torch.nn.ModuleList([Qwen3DFlash2DecoderLayer(cfg,i,block_size=8,conv_kernel_size=2,conv_group_size=16) for i in range(2)])
    for layer in layers:layer.reset_convolutions()
    layers=layers.to(device=device,dtype=dtype).train();copied=copy.deepcopy(layers)
    for layer in copied:
        layer.gradient_checkpointing=True
        layer._gradient_checkpointing_func=functools.partial(checkpoint,use_reentrant=False)
    x=torch.randn(1,16,64,device=device,dtype=dtype)
    target=torch.randn(1,32,64,device=device,dtype=dtype)
    mask=torch.zeros(1,1,16,48,device=device,dtype=dtype);mask[:,:,:,-5:]=float('-inf')
    cos=torch.ones(1,48,16,device=device,dtype=dtype);sin=torch.zeros_like(cos)
    outputs=[];gradients=[]
    for model in [layers,copied]:
        q=x.detach().clone().requires_grad_();t=target.detach().clone().requires_grad_()
        for layer in model:q=layer(hidden_states=q,target_hidden=t,attention_mask=mask,position_embeddings=(cos,sin))
        loss=q.float().square().mean();loss.backward();outputs.append(loss.item())
        gradients.append(torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None]))
    a,b=gradients;err=(a-b).abs().max().item();rel=((a-b).square().sum()/a.square().sum().clamp_min(1e-30)).sqrt().item()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert abs(outputs[0]-outputs[1])<(1e-6 if device=='cpu' else 5e-4)
    assert rel<(1e-5 if device=='cpu' else .02),(device,err,rel)
    return {'device':device,'dtype':str(dtype),'losses':outputs,'gradient_max_error':err,'gradient_relative_l2':rel,'passed':True}

if __name__=='__main__':
    torch.npu.set_compile_mode(jit_compile=False)
    result={'cpu':check('cpu',torch.float32),'npu':check('npu:0',torch.bfloat16),'use_reentrant':False}
    pathlib.Path(os.environ['AUDIT_OUT']).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
