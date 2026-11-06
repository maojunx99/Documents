"""Native HTTP/shared-file integration, no authentication/gateway or accelerator jobs."""
import concurrent.futures
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.managed import same,write
from teachers.autodeploy import nodes
from teachers.launch import npu_runtime_devices
from teachers.client import RemoteTeachers
from teachers.protocol import load_peers,tensor_file

spec=importlib.util.spec_from_file_location('atomic_shared_file',ROOT/'npu/source/hs_connectors/src/hs_connectors/atomic_file.py')
atomic=importlib.util.module_from_spec(spec);spec.loader.exec_module(atomic)


def fixture(path,tokens,layers=2,width=4):
    count=len(tokens)*layers*width*2
    header={'hidden_states':{'dtype':'BF16','shape':[len(tokens),layers,width],'data_offsets':[0,count]},
            'token_ids':{'dtype':'I64','shape':[len(tokens)],'data_offsets':[count,count+8*len(tokens)]}}
    raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
    Path(path).write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*count+struct.pack('<'+'q'*len(tokens),*tokens))


class NativeTeachers(unittest.TestCase):
    def test_node_groups_and_invalid_assignment(self):
        result=nodes(['local|4,5|2','npu-b|8,9,10,11|2|old-container'],'default',[0,1,2,3])
        self.assertEqual([v['cards'] for v in result],[[4,5],[8,9],[10,11]])
        for value in [['local|0,1|2'],['npu|0,0|2'],['npu|0,1,2|2'],['npu|0,1|2','npu|1,2|2'],['npu;bad|0,1|2']]:
            with self.assertRaises(ValueError):nodes(value,'default',[0])

    def test_physical_npu_mapping(self):
        class Runtime:
            def get_logic_dev_id_by_phy_dev_id(self,x):return (x-8,0) if 8<=x<=11 else (-1,1)
            def get_phy_dev_id_by_logic_dev_id(self,x):return (x+8,0)
        self.assertEqual(npu_runtime_devices([10,11],Runtime()),'2,3')
        with self.assertRaises(ValueError):npu_runtime_devices([0],Runtime())

    def test_plain_endpoint_config_has_no_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'peers.json'
            peer={'name':'a','url':'http://teacher.internal:8395','shared_root':str(Path(tmp).resolve())}
            write(path,{'teachers':[peer]});self.assertEqual(load_peers(path),[peer])
            peer['api_key_file']='obsolete'
            write(path,{'teachers':[peer]})
            with self.assertRaises(ValueError):load_peers(path)

    def test_atomic_publication_and_failed_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.safetensors'
            started=threading.Event();release=threading.Event()
            def save(temp):
                Path(temp).write_bytes(b'partial');started.set();release.wait(5);fixture(temp,[1,2,3])
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(atomic.publish,path,save);self.assertTrue(started.wait(5))
                self.assertFalse(path.exists());release.set();future.result(timeout=5)
            self.assertEqual(tensor_file(atomic.wait(path),2,4,[1,2,3],finite=True)['tokens'],3)
            failed=Path(tmp)/'bad.safetensors'
            def bad(_):raise ValueError('intentional writer error')
            with self.assertRaises(ValueError):atomic.publish(failed,bad)
            with self.assertRaisesRegex(RuntimeError,'intentional'):atomic.wait(failed,timeout=.1)
            self.assertFalse(failed.exists())

    def test_direct_native_http_two_teachers_and_completed_file_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();cache=root/'cache';cache.mkdir()
            write(model/'config.json',{'hidden_size':4});headers=[];threads=[];servers=[]
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*_):pass
                def do_GET(self):
                    data=b'' if self.path=='/health' else json.dumps({'data':[{'id':str(model),'max_model_len':8192}]}).encode()
                    self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                def do_POST(self):
                    headers.append(self.headers.get('Authorization'))
                    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    tokens=body['prompt'];path=cache/(__import__('uuid').uuid4().hex+'.safetensors')
                    def save(temp):time.sleep(.03);fixture(temp,tokens)
                    thread=threading.Thread(target=atomic.publish,args=(path,save));thread.start();threads.append(thread)
                    data=json.dumps({'choices':[{'prompt_token_ids':tokens}],'kv_transfer_params':{'hidden_states_path':str(path)}}).encode()
                    self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            try:
                peers=[]
                for i in range(2):
                    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);servers.append(server)
                    threading.Thread(target=server.serve_forever,daemon=True).start()
                    peers.append({'name':f't{i}','url':f'http://127.0.0.1:{server.server_port}','shared_root':str(cache)})
                client=RemoteTeachers(peers,model,'npu',[1],root/'receipts')
                self.assertEqual(client.start(),peers);client.check();client.close()
                self.assertEqual(headers,[None]*4)
                self.assertFalse(list(cache.iterdir()))
                self.assertTrue(json.loads((root/'receipts/ADMISSION.json').read_text())['passed'])
            finally:
                for server in servers:server.shutdown();server.server_close()
                for thread in threads:thread.join(timeout=5)

    @unittest.skipUnless(os.name=='posix','Linux process identity checks')
    def test_native_exec_cleanup_preserves_unrelated_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);agent=[sys.executable,str(ROOT/'teachers/node_agent.py')]
            sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])
            command=[sys.executable,'-c',"import os,sys;os.execv(sys.executable,[sys.executable,'-c','import time;time.sleep(90)'])"]
            write(root/'request.json',{'argv':command,'service_run':str(root/'unused')})
            try:
                subprocess.run(agent+['start','--control',str(root)],check=True,capture_output=True)
                time.sleep(.1)
                subprocess.run(agent+['check','--control',str(root)],check=True,capture_output=True)
                subprocess.run(agent+['stop','--control',str(root)],check=True,capture_output=True)
                self.assertTrue(json.loads((root/'STOPPED.json').read_text())['passed'])
                self.assertIsNone(sentinel.poll())
            finally:sentinel.terminate();sentinel.wait(timeout=5)

    def test_gpu_npu_atomic_transport_code_is_identical(self):
        for name in ['atomic_file.py','shared_file.py','shared_file_connector.py']:
            self.assertEqual((ROOT/'npu/source/hs_connectors/src/hs_connectors'/name).read_bytes(),
                             (ROOT/'gpu/source/hs_connectors/src/hs_connectors'/name).read_bytes())

    def test_gpu_remote_only_plan_uses_native_shared_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);peers=root/'peers.json';run=root/'new_run'
            write(peers,{'teachers':[{'name':'a','url':'http://teacher.internal:8395',
                                     'shared_root':str(run/'hidden_states')}]})
            result=subprocess.run([sys.executable,str(ROOT/'gpu/run_pipeline.py'),'--model',str(root/'model'),
                '--data-dir',str(root/'data'),'--run',str(run),'--teacher-devices','',
                '--remote-teachers',str(peers),'--dry-run'],check=True,capture_output=True,text=True)
            plan=json.loads(result.stdout)
            self.assertFalse(plan['teachers'])
            self.assertEqual(plan['train'][plan['train'].index('--hidden-states-backend')+1],'shared_file')
            self.assertFalse(run.exists())


if __name__=='__main__':unittest.main()
