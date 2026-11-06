"""Direct native vLLM admission; shared files carry features, no gateway/token/lease."""
import json
from pathlib import Path
import time
import urllib.request

from common.managed import write
from teachers.protocol import tensor_file


def rpc(peer, path, body=None, timeout=15):
    req = urllib.request.Request(peer['url']+path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read(16*1024**2)
        return json.loads(content) if content else {}


class RemoteTeachers:
    def __init__(self, peers, model, backend, target_layers, folder, startup_timeout=1200, required_context=1):
        self.peers, self.model, self.folder = peers, Path(model), Path(folder)
        self.target_layers = target_layers
        self.required_context, self.startup_timeout = required_context, startup_timeout
        self.last_check = 0

    def check(self):
        if time.monotonic()-self.last_check < 15:
            return
        for peer in self.peers:
            rpc(peer, '/health', timeout=5)
        self.last_check = time.monotonic()

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        raw = json.loads((self.model/'config.json').read_text()); text = raw.get('text_config',raw)
        for peer in self.peers:
            deadline = time.monotonic()+self.startup_timeout
            while True:
                try:
                    rpc(peer, '/health', timeout=3)
                    break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError('Native teacher readiness: '+peer['name'])
                    time.sleep(1)
            models = rpc(peer, '/v1/models')
            selected = [m for m in models['data'] if m['id']==str(self.model)]
            if not selected or selected[0].get('max_model_len', self.required_context)<self.required_context:
                raise ValueError('Teacher served-model name/context differs from the selected target')
            root = Path(peer['shared_root']).resolve()
            probes = []
            for length in [32,256]:
                tokens = [1]*length
                result = rpc(peer, '/v1/completions', {'model':str(self.model), 'prompt':tokens,
                    'max_tokens':1, 'return_token_ids':True}, timeout=910)
                params = result.get('kv_transfer_params',{})
                handle = Path(params.get('hidden_states_path') or params.get('handle',''))
                if handle.is_symlink() or handle.parent.resolve()!=root or handle.suffix!='.safetensors':
                    raise ValueError('Native teacher returned a handle outside the shared job cache')
                deadline = time.monotonic()+900
                while not handle.exists():
                    error = Path(str(handle)+'.failed')
                    if error.exists():
                        raise RuntimeError('Native shared-file writer failed: '+str(error))
                    if time.monotonic()>deadline:
                        raise TimeoutError('Teacher file is not visible on the trainer shared mount')
                    time.sleep(.02)
                proof = tensor_file(handle,len(self.target_layers)+1,text['hidden_size'],tokens,finite=True)
                if result['choices'][0]['prompt_token_ids']!=tokens:
                    raise ValueError('Native teacher response token IDs mismatch')
                handle.unlink(); probes.append(proof)
            write(self.folder/(peer['name']+'_admission.json'), {'passed':True,'url':peer['url'],
                'probes':probes,'shared_root':str(root),'native_http_direct':True,'authentication':False,
                'transport':'shared_file; atomic writer publication; no cross-host flock'})
        write(self.folder/'ADMISSION.json',{'passed':True,'teachers':[p['name'] for p in self.peers]})
        return list(self.peers)

    def close(self, consumers_stopped=True):
        # Process ownership belongs to autodeploy/OwnedProcesses, never to an arbitrary URL.
        write(self.folder/'CLEANUP.json',{'passed':consumers_stopped,'consumers_stopped':consumers_stopped,
              'process_cleanup':'Only automatically launched processes are stopped by Deployment.'})


def connect(peers, model, backend, target_layers, folder, required_context=1):
    client = RemoteTeachers(peers,model,backend,target_layers,folder,required_context=required_context)
    return client, client.start()


def router_environment(folder, local_ports, remote_backends):
    backends = [{'name':'local_'+str(p),'url':f'http://127.0.0.1:{p}'} for p in local_ports]+remote_backends
    if not backends:
        raise ValueError('At least one teacher is required')
    path = Path(folder)/'ROUTER_BACKENDS.json'; write(path,{'backends':backends})
    return {'TEACHER_BACKENDS_FILE':str(path)}
