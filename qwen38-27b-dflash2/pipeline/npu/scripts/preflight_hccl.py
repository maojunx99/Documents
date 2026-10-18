import json, os, pathlib
import torch
import torch_npu
import torch.distributed as dist

rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
world=int(os.environ['WORLD_SIZE']);expected=world*(world+1)/2
torch.npu.set_device(local);torch.npu.set_compile_mode(jit_compile=False)
dist.init_process_group('hccl',device_id=torch.device('npu',local))
scalar=torch.tensor(float(rank+1),device=f'npu:{local}')
dist.reduce(scalar,dst=0)
if rank==0:assert scalar.item()==expected
stacked=torch.tensor([float(rank+1),1.],device=f'npu:{local}')
dist.all_reduce(stacked);assert stacked.tolist()==[expected,float(world)]
weight=torch.tensor([float(rank)],device=f'npu:{local}');dist.broadcast(weight,src=0);assert weight.item()==0
pathlib.Path(os.environ['AUDIT_OUT']+f'.rank{rank}.json').write_text(json.dumps({'rank':rank,'world_size':world,'backend':dist.get_backend(),'scalar_reduce':True,'all_reduce':True,'broadcast':True}))
dist.barrier();dist.destroy_process_group()
