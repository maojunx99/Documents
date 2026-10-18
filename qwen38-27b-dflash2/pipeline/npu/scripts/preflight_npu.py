"""Real NPU forward/backward parity checks before running the full model."""
import json, os, pathlib, time
import torch
import torch_npu
import torch.nn.functional as F

def main():
    from speculators.losses import fused, eager
    torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(42);torch.npu.manual_seed_all(42)
    evidence={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device':torch.npu.get_device_name(0),'pid':os.getpid(),'starttime':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
    for name in ['ce','tv']:
        x=torch.randn(1,8,248320,device='npu',dtype=torch.float32,requires_grad=True)
        y=torch.randn_like(x)
        fn=getattr(fused,'fused_'+name+'_loss');ref=getattr(eager,name+'_loss')
        actual=fn(x,y);ga=torch.autograd.grad(actual.sum(),x)[0]
        expected=ref(x,y);ge=torch.autograd.grad(expected.sum(),x)[0]
        loss_error=(actual-expected).abs().max().item();grad_error=(ga-ge).abs().max().item()
        assert torch.isfinite(actual).all() and torch.isfinite(ga).all()
        assert loss_error<2e-4 and grad_error<2e-5,(name,loss_error,grad_error)
        evidence[name]={'loss_max_error':loss_error,'gradient_max_error':grad_error,'finite':True}
        del x,y,actual,expected,ga,ge
    q=torch.randn(1,4,32,128,device='npu',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(1,4,64,128,device='npu',dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.zeros(1,1,32,64,device='npu',dtype=torch.bfloat16);mask[:,:,:,-5:]=float('-inf')
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    out.float().square().mean().backward()
    reference=torch.softmax(q.float()@k.float().transpose(-1,-2)/(128**.5)+mask.float(),-1)@v.float()
    error=(out.float()-reference).abs().max().item()
    assert error<.02 and all(t.grad is not None and torch.isfinite(t.grad).all() for t in [q,k,v])
    evidence['sdpa']={'max_error':error,'finite_backward':True}
    evidence['passed']=True
    pathlib.Path(os.environ['AUDIT_OUT']).write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence),flush=True)

if __name__=='__main__':main()
