"""CPU HTTP/file/lifecycle integration; fake native teacher, no accelerator jobs."""
import concurrent.futures
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.managed import write
from teachers.client import RemoteTeachers, router_environment, rpc
from teachers.protocol import (VERSION, digest, load_peers, model_manifest, source_manifest,
                               runtime_manifest, tensor_file, publish_complete)
from teachers.service import Engine, make_server


def fixture_tensor(path, tokens, layers=2, width=4, bad_finite=False):
    hidden = len(tokens)*layers*width*2
    header = {'hidden_states': {'dtype':'BF16','shape':[len(tokens),layers,width],'data_offsets':[0,hidden]},
              'token_ids': {'dtype':'I64','shape':[len(tokens)],'data_offsets':[hidden,hidden+8*len(tokens)]}}
    data = json.dumps(header,separators=(',',':')).encode()
    data += b' '*((-len(data))%8)
    values = (b'\x80\x7f' if bad_finite else b'\x00\x00')*(hidden//2)
    Path(path).write_bytes(struct.pack('<Q',len(data))+data+values+struct.pack('<'+'q'*len(tokens),*tokens))


class FakeEngine(Engine):
    def start_native(self, served_name):
        with self.lock:
            self.phase='ready'


class Harness:
    def __init__(self, root, name, model, key_file):
        self.run=root/name;self.run.mkdir()
        for folder in ['staging','ready']:(self.run/folder).mkdir()
        write(self.run/'OWNER.json',{'run':str(self.run)})
        self.requests=[]
        owner=self
        class Native(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_GET(self):
                data=json.dumps({'healthy':True}).encode();self.send_response(200)
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                payload=self.rfile.read(int(self.headers['Content-Length']))
                body=json.loads(payload);owner.requests.append(payload)
                tokens=body['prompt']
                path=owner.run/'staging'/(__import__('uuid').uuid4().hex+'.safetensors')
                fixture_tensor(path,tokens)
                reply={'choices':[{'prompt_token_ids':tokens}], 'kv_transfer_params':{'hidden_states_path':str(path)}}
                data=json.dumps(reply).encode();self.send_response(200)
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.native=ThreadingHTTPServer(('127.0.0.1',0),Native)
        threading.Thread(target=self.native.serve_forever,daemon=True).start()
        self.a=SimpleNamespace(run=self.run,model=model,backend='npu',api_key_file=key_file,
                native_port=self.native.server_port,target_layer_ids=[0],max_model_len=4096,
                request_timeout=2,lease_seconds=15,idle_timeout=120)
        contract={'protocol':VERSION,'backend':'npu','target':model_manifest(model),'source':source_manifest('npu'),
                  'runtime':runtime_manifest('npu'),'target_layer_ids':[0],'hidden_size':4,'shared_root':str(self.run)}
        self.engine=FakeEngine(self.a,contract)
        self.server=make_server(('127.0.0.1',0),self.engine)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.peer={'name':name,'url':f'http://127.0.0.1:{self.server.server_port}',
                   'api_key_file':str(key_file),'shared_root':str(self.run)}
    def close(self):
        if not self.engine.finished.is_set():self.engine.stop('expired',cleanup=False)
        self.server.shutdown();self.server.server_close()
        self.native.shutdown();self.native.server_close()


class RemoteTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'hidden_size':4,'vocab_size':256,'num_hidden_layers':2}))
        (self.model/'model.safetensors').write_bytes(b'fake-native-teacher-does-not-load-these-weights')
        self.key=self.root/'key';self.key.write_text('a'*64);self.key.chmod(0o600)
        self.services=[]
    def tearDown(self):
        for service in self.services:service.close()
        self.temp.cleanup()
    def service(self,name='teacher_a'):
        h=Harness(self.root,name,self.model,self.key);self.services.append(h);return h
    def client(self,peers,folder='client',**kwargs):
        return RemoteTeachers(peers,self.model,'npu',[0],self.root/folder,**kwargs)

    def test_atomic_publication_rejects_partial_wrong_tokens_and_foreign_paths(self):
        staging=self.root/'staging';staging.mkdir();ready=self.root/'ready';ready.mkdir()
        p=staging/'feature.safetensors';fixture_tensor(p,[3,4])
        before=digest(p)
        with self.assertRaisesRegex(ValueError,'token IDs'):
            publish_complete(p,staging,ready,2,4,[4,3])
        self.assertFalse(list(ready.iterdir()));self.assertTrue(p.exists())
        with self.assertRaisesRegex(ValueError,'owned staging'):
            publish_complete(p,self.root/'other',ready,2,4,[3,4])
        content=p.read_bytes();p.write_bytes(content[:-1])
        with self.assertRaisesRegex(ValueError,'payload'):
            publish_complete(p,staging,ready,2,4,[3,4])
        p.write_bytes(content)
        handle,receipt=publish_complete(p,staging,ready,2,4,[3,4])
        self.assertEqual(digest(handle),before);self.assertFalse(p.exists())
        self.assertEqual(receipt['tokens'],2)

    def test_finite_probe_rejects_nonfinite_bf16(self):
        p=self.root/'bad.safetensors';fixture_tensor(p,[1],bad_finite=True)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            tensor_file(p,2,4,[1],finite=True)

    def test_peer_config_rejects_credentials_duplicates_and_bad_keys(self):
        h=self.service();path=self.root/'peers.json'
        write(path,{'teachers':[h.peer]});self.assertEqual(load_peers(path)[0]['url'],h.peer['url'])
        write(path,{'teachers':[h.peer,h.peer]})
        with self.assertRaisesRegex(ValueError,'Duplicate'):load_peers(path)
        bad=dict(h.peer,url='http://user:password@teacher.internal:1234')
        write(path,{'teachers':[bad]})
        with self.assertRaisesRegex(ValueError,'credentials'):load_peers(path)

    def test_authentication_and_identity_failure_start_no_teacher(self):
        h=self.service()
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(h.peer['url']+'/__teacher/info')
        self.assertEqual(e.exception.code,401)
        contract=copy.deepcopy(h.engine.contract);contract['hidden_size']=5
        with self.assertRaises(urllib.error.HTTPError):
            rpc(h.peer,'/__teacher/claim',{'run_id':'1'*32,'contract':contract,'served_model':str(self.model)})
        self.assertEqual(h.engine.phase,'idle');self.assertFalse(h.requests)
        self.assertFalse((h.run/'CLAIM.json').exists())
        client=self.client([h.peer],required_context=8192)
        with self.assertRaisesRegex(ValueError,'context'):client.start()
        client.close();self.assertEqual(h.engine.phase,'idle')

    def test_two_teacher_admission_router_concurrency_and_cleanup(self):
        first,second=self.service('teacher_a'),self.service('teacher_b')
        client=self.client([first.peer,second.peer])
        backends=client.start()
        self.assertEqual([len(x.requests) for x in self.services],[2,2])
        environment=router_environment(self.root,[],backends)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ,**environment,TEACHER_ROUTER_PORT=str(port),
                 TEACHER_ROUTING_LOG=str(self.root/'router.jsonl'))
        proc=subprocess.Popen([sys.executable,str(ROOT/'gpu/teacher_router.py')],env=env,
                               stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        handles=[]
        try:
            deadline=time.monotonic()+10
            while True:
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1).close();break
                except OSError:
                    if proc.poll() is not None or time.monotonic()>deadline:raise
                    time.sleep(.05)
            def one(i):
                tokens=[1,i+2,7]
                payload=json.dumps({'model':str(self.model),'prompt':tokens,'max_tokens':1,
                                    'return_token_ids':True}).encode()
                request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=payload,
                                               headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=10) as r:result=json.load(r)
                handle=Path(result['kv_transfer_params']['hidden_states_path'])
                tensor_file(handle,2,4,tokens,finite=True)
                return handle,payload
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                rows=list(pool.map(one,range(8)))
            handles=[row[0] for row in rows]
            self.assertEqual(len(set(handles)),8)
            for _,payload in rows:self.assertIn(payload,first.requests+second.requests)
            self.assertGreater(len(first.requests),2);self.assertGreater(len(second.requests),2)
            client.check()
        finally:
            proc.terminate();proc.communicate(timeout=10)
            client.close(consumers_stopped=True)
        for h in self.services:
            receipt=json.loads((h.run/'CLEANUP.json').read_text())
            self.assertTrue(receipt['passed']);self.assertTrue(receipt['cache_removed'])
            self.assertEqual(receipt['run_id'],client.run_id)
            self.assertFalse((h.run/'ready').exists())
        self.assertTrue(json.loads((client.folder/'CLEANUP.json').read_text())['passed'])
        self.assertNotIn(self.key.read_text().strip(),''.join(p.read_text() for p in client.folder.glob('*.json')))

    def test_rejected_second_claim_cannot_release_first_owner(self):
        h=self.service();first=self.client([h.peer],'first');first.start()
        second=self.client([h.peer],'second')
        with self.assertRaises(urllib.error.HTTPError):second.start()
        second.close()
        self.assertEqual(h.engine.run_id,first.run_id);self.assertEqual(h.engine.phase,'ready')
        first.close()

    def test_lease_expiry_preserves_inflight_cache(self):
        h=self.service();client=self.client([h.peer]);client.start()
        client.stop_event.set();client.thread.join(timeout=5)
        p=h.run/'ready/keep.safetensors';fixture_tensor(p,[1])
        h.engine.last_seen=time.monotonic()-20
        worker=threading.Thread(target=h.engine.watchdog,daemon=True);worker.start()
        self.assertTrue(h.engine.finished.wait(5));worker.join(timeout=1)
        receipt=json.loads((h.run/'CLEANUP.json').read_text())
        self.assertEqual(receipt['reason'],'expired');self.assertTrue(p.exists())
        self.assertFalse(receipt['cache_removed'])

    def test_npu_remote_only_and_mixed_plans(self):
        from npu.recipe import parser,validate,command_plan
        model=self.root/'large';model.mkdir()
        (model/'config.json').write_text(json.dumps({'text_config':{
            'hidden_size':5120,'vocab_size':248320,'num_hidden_layers':64}}))
        key=self.root/'peers.json'
        peer={'name':'teacher_a','url':'http://teacher-a.internal:8395','api_key_file':str(self.key),
              'shared_root':str(self.root/'teacher_a')}
        write(key,{'teachers':[peer]})
        for groups,count in [('',0),('0,1;2,3',2)]:
            a=parser().parse_args(['--model',str(model),'--run',str(self.root/'new'),
                '--input',str(self.root/'data.jsonl'),'--teacher-groups',groups,'--remote-teachers',str(key)])
            resolved=validate(a);plan=command_plan(a,resolved)
            self.assertEqual(len(plan['teachers']),count)
            self.assertEqual(len(plan['remote_teachers']),1)
            self.assertNotIn('Bearer',json.dumps(plan))
            self.assertFalse(a.run.exists())
        a.remote_teachers=None;a.teacher_groups=''
        with self.assertRaisesRegex(ValueError,'At least one'):validate(a)
        a.remote_teachers=key;a.stage='evaluate';self.key.unlink()
        self.assertEqual(len(validate(a)['remote_teachers']),1)  # Completed-model evaluation needs no teacher secret.

    def test_target_change_blocks_clean_completion(self):
        h=self.service();client=self.client([h.peer]);client.start()
        (self.model/'model.safetensors').write_bytes(b'changed-weights')
        with self.assertRaisesRegex(RuntimeError,'cleanup'):client.close()
        receipt=json.loads((h.run/'CLEANUP.json').read_text())
        self.assertFalse(receipt['passed']);self.assertTrue((h.run/'ready').exists())

    def test_gpu_remote_only_dry_run_keeps_four_trainers_and_starts_no_processes(self):
        config=self.root/'peers.json'
        write(config,{'teachers':[{'name':'teacher_a','url':'http://teacher.internal:8395',
              'api_key_file':str(self.key),'shared_root':str(self.root/'shared')}]})
        run=self.root/'gpu_new'
        p=subprocess.run([sys.executable,str(ROOT/'gpu/run_pipeline.py'),'--model',str(self.model),
              '--data-dir',str(self.root/'shards'),'--run',str(run),'--teacher-devices','',
              '--train-devices','0,1,2,3','--remote-teachers',str(config),'--dry-run'],
              capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        plan=json.loads(p.stdout);self.assertFalse(plan['teachers'])
        self.assertEqual(len(plan['remote_teachers']),1)
        self.assertEqual(plan['train'][plan['train'].index('--nproc_per_node')+1],'4')
        self.assertFalse(run.exists())

    @unittest.skipUnless(os.name=='posix','Writer flock requires Linux')
    def test_publisher_waits_for_the_producer_local_lock(self):
        staging=self.root/'staging';staging.mkdir();ready=self.root/'ready';ready.mkdir()
        path=staging/'async.safetensors';fixture_tensor(path,[1,2,3])
        import fcntl
        lock=Path(str(path)+'.lock')
        with lock.open('w') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            results=[]
            worker=threading.Thread(target=lambda:results.append(publish_complete(path,staging,ready,2,4,[1,2,3])))
            worker.start();time.sleep(.15)
            self.assertFalse(results);self.assertFalse(list(ready.iterdir()))
            fcntl.flock(handle,fcntl.LOCK_UN)
            worker.join(timeout=5)
        self.assertEqual(len(results),1);self.assertTrue(Path(results[0][0]).exists())

    @unittest.skipUnless(os.name=='posix' and Path('/proc').exists(),'Guardian process identities require Linux')
    def test_guardian_owner_death_releases_only_registered_cpu_teacher(self):
        from common.managed import same
        job=self.root/'guardian_job';job.mkdir();(job/'owned').mkdir()
        code="""import os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from common.managed import identity,write,OwnedProcesses
root=Path(sys.argv[2]);write(root/'OWNER.json',{'run':str(root),'identity':identity(os.getpid())})
owner=OwnedProcesses(root/'owned')
owner.start('teacher',[sys.executable,'-c','import time;time.sleep(120)'],dict(os.environ))
time.sleep(120)
"""
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
        owner=subprocess.Popen([sys.executable,'-c',code,str(ROOT),str(job)],start_new_session=True)
        guardian=None;teacher=None
        try:
            deadline=time.monotonic()+10
            while not (job/'owned/teacher_identity.json').exists():
                if owner.poll() is not None or time.monotonic()>deadline:raise RuntimeError('CPU owner startup')
                time.sleep(.02)
            teacher=json.loads((job/'owned/teacher_identity.json').read_text())
            guardian=subprocess.Popen([sys.executable,str(ROOT/'teachers/guardian.py'),'--run',str(job)])
            owner.kill();owner.wait(timeout=5);guardian.wait(timeout=40)
            receipt=json.loads((job/'GUARDIAN_CLEANUP.json').read_text())
            self.assertTrue(receipt['passed']);self.assertFalse(same(teacher))
            self.assertIsNone(sentinel.poll());self.assertTrue((job/'owned').exists())
        finally:
            if owner.poll() is None:owner.kill();owner.wait(timeout=5)
            if guardian and guardian.poll() is None:guardian.terminate();guardian.wait(timeout=5)
            if teacher and same(teacher):os.kill(teacher['pid'],__import__('signal').SIGKILL)
            sentinel.terminate();sentinel.wait(timeout=5)


if __name__=='__main__':unittest.main()
