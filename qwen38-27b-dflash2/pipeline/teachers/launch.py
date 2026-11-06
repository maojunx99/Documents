"""Launch the native teacher directly in an existing environment; no HTTP gateway."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.managed import assert_free, identity, source_environment, write
from teachers.protocol import digest


def npu_runtime_devices(cards, runtime=None):
    if runtime is None:
        import acl
        runtime=acl.rt
    result=[]
    for physical in cards:
        logical,ret=runtime.get_logic_dev_id_by_phy_dev_id(physical)
        if ret:raise ValueError(f'Physical NPU {physical} is unavailable in this container: {ret}')
        check,ret=runtime.get_phy_dev_id_by_logic_dev_id(logical)
        if ret or check!=physical:raise ValueError('NPU mapping did not round trip')
        result.append(logical)
    return ','.join(map(str,result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--backend',choices=['npu','gpu'],default='npu')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--devices',required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--hidden-states-path',type=Path,required=True)
    p.add_argument('--served-model-name',required=True)
    p.add_argument('--bind',default='0.0.0.0')
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--distributed-port',type=int,required=True)
    p.add_argument('--target-layer-ids',type=int,nargs='+',default=[6,20,34,48,62])
    p.add_argument('--max-model-len',type=int,default=8192)
    p.add_argument('--max-num-seqs',type=int,default=4)
    p.add_argument('--memory-utilization',type=float,default=.8)
    a=p.parse_args();a.model=a.model.resolve();a.run=a.run.resolve()
    if a.dry_run:
        print(json.dumps({'backend':a.backend,'model':str(a.model),'devices':a.devices,'transport':'shared_file',
              'native_http_direct':True,'authentication':False,'extra_gateway':False}));return
    a.run.mkdir(parents=True,exist_ok=False)
    try:
        cards=[int(x) for x in a.devices.split(',')]
        if len(cards)!=len(set(cards)) or min(cards)<0:raise ValueError('Invalid teacher cards')
        assert_free(a.backend,cards)
        cache=a.hidden_states_path.resolve()
        if not cache.is_dir() or not (cache/'OWNER.json').exists():
            raise ValueError('Shared cache must be created and owned by the trainer run first')
        # This marker must be visible on the trainer before native admission.
        probe=os.urandom(4096);probe_path=cache/(a.run.name+'_mount_probe.bin')
        with probe_path.open('xb') as f:f.write(probe);f.flush();os.fsync(f.fileno())
        write(a.run/'SHARED_PROBE.json',{'path':str(probe_path),'sha256':digest(probe_path)})
        target={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,
                       'sha256':digest(f) if f.stat().st_size<1024**2 else None}
                for f in a.model.iterdir() if f.is_file()}
        write(a.run/'TARGET_IDENTITY.json',target)
        source=ROOT/a.backend/'source';env=source_environment(source)
        env['PYTHONPATH']=os.pathsep.join([str(ROOT/'npu/scripts') if a.backend=='npu' else str(ROOT),env.get('PYTHONPATH','')])
        visible=a.devices if a.backend=='gpu' else npu_runtime_devices(cards)
        env['CUDA_VISIBLE_DEVICES' if a.backend=='gpu' else 'ASCEND_RT_VISIBLE_DEVICES']=visible
        env['VLLM_PORT']=str(a.distributed_port)
        write(a.run/'DEVICE_MAPPING.json',{'physical_cards':cards,'runtime_devices':visible})
        command=[sys.executable,str(source/'scripts/launch_vllm.py'),str(a.model),
          '--target-layer-ids',*map(str,a.target_layer_ids),'--hidden-states-backend','shared_file',
          '--hidden-states-path',str(cache)]
        if a.backend=='gpu':command+=['--provenance-dir',str(a.run/'native_provenance')]
        command+=['--','--host',a.bind,'--port',str(a.port),'--served-model-name',a.served_model_name,
          '--tensor-parallel-size',str(len(cards)),'--dtype','bfloat16','--enforce-eager',
          '--no-enable-chunked-prefill','--gpu-memory-utilization',str(a.memory_utilization),
          '--max-num-seqs',str(a.max_num_seqs),'--max-model-len',str(a.max_model_len),
          '--max-num-batched-tokens',str(a.max_model_len)]
        write(a.run/'teacher_provenance.json',{'argv':command,'identity':identity(os.getpid()),
              'transport':'native shared_file atomic publication','native_http_direct':True,
              'wrapper_consumed_provenance':a.backend=='npu'})
        os.execve(command[0],command,env)
    except BaseException:
        write(a.run/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
