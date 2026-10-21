"""CPU contract tests: model switches, topology guards, sampling, and conversation coverage."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'regeneration'))
from npu.recipe import parser,validate,command_plan
from verify_outputs import verify


class PipelineContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.model=self.root/'Qwen3.6-27B'
        self.model.mkdir()
        self.target={'hidden_size':5120,'vocab_size':248320,'num_hidden_layers':64}
        (self.model/'config.json').write_text(json.dumps({'text_config':self.target}))

    def tearDown(self):self.temp.cleanup()

    def args(self,*extra):
        return parser().parse_args(['--model',str(self.model),'--run',str(self.root/'new_run'),
                                    '--input',str(self.root/'prompts.jsonl'),*extra])

    def test_configurable_mesh(self):
        for x,y in [(1,4),(2,2),(2,4),(4,2),(4,4)]:
            cards=','.join(map(str,range(8,8+x*y)))
            a=self.args('--replicas',str(x),'--shard-size',str(y),'--train-devices',cards)
            result=validate(a);plan=command_plan(a,result)
            self.assertEqual(result['mesh'],[list(range(r*y,(r+1)*y)) for r in range(x)])
            trainer=plan['trainer']
            self.assertEqual(trainer[trainer.index('--nproc_per_node')+1],str(x*y))
            self.assertEqual(trainer[trainer.index('--parallel-replicas')+1],str(x))
            self.assertEqual(trainer[trainer.index('--parallel-shard-size')+1],str(y))

    def test_invalid_topology_fails_before_any_job(self):
        for options in [('--replicas','1'),('--train-devices','0,0'),('--teacher-groups','8,9'),
                        ('--regeneration-groups','8,9;9,10'),('--shard-size','0')]:
            with self.assertRaises(ValueError):validate(self.args(*options))

    def test_epoch0_stop_preserves_explicit_five_epoch_schedule(self):
        a=self.args('--epochs','5','--stop-after-epoch','0','--scheduler-total-steps','258565',
                    '--scheduler-warmup-steps','10342','--replicas','4','--shard-size','2')
        plan=command_plan(a,validate(a))['trainer']
        for key,value in [('--epochs','5'),('--parallel-stop-after-epoch','0'),
                          ('--scheduler-total-steps','258565'),('--scheduler-warmup-steps','10342')]:
            self.assertEqual(plan[plan.index(key)+1],value)
        self.assertNotIn('--max-steps',plan)
        for options in [('--stop-after-epoch','5'),('--stop-after-epoch','-1'),
                        ('--scheduler-total-steps','0'),('--scheduler-warmup-steps','-1'),
                        ('--scheduler-total-steps','100','--scheduler-warmup-steps','100')]:
            with self.assertRaises(ValueError):validate(self.args(*options))

    def test_new_target_and_sampling_propagate_to_all_stages(self):
        a=self.args('--temperature','.8','--top-p','.9','--top-k','40','--presence-penalty','0',
                    '--reasoning-effort','high','--generation-seed','123','--max-steps','2')
        plan=command_plan(a,validate(a))
        for name in ['generation_server','regeneration','prepare','trainer']:
            self.assertIn(str(self.model),plan[name])
        for teacher in plan['teachers']:self.assertIn(str(self.model),teacher)
        regen=plan['regeneration']
        for key,value in [('--temperature','0.8'),('--top-p','0.9'),('--top-k','40'),
                          ('--presence-penalty','0.0'),('--reasoning-effort','high'),('--generation-seed','123')]:
            self.assertEqual(regen[regen.index(key)+1],value)
        self.assertEqual(plan['trainer'][-2:],['--max-steps','2'])

    def test_structure_mismatch_is_rejected(self):
        for key in ['hidden_size','vocab_size']:
            target=copy.deepcopy(self.target);target[key]+=1
            (self.model/'config.json').write_text(json.dumps(target))
            with self.assertRaisesRegex(ValueError,key):validate(self.args())

    def test_dry_run_needs_no_accelerator_and_creates_no_run(self):
        result=subprocess.run([sys.executable,str(ROOT/'npu/run_pipeline.py'),
            '--model',str(self.model),'--run',str(self.root/'not_created'),
            '--input',str(self.root/'not_required_for_planning.jsonl'),'--dry-run'],capture_output=True,text=True,check=True)
        payload=json.loads(result.stdout)
        self.assertEqual(payload['rank_mesh'],[[0,1,2,3],[4,5,6,7]])
        self.assertFalse((self.root/'not_created').exists())

    def test_conversation_fanout_and_sampling_integrity(self):
        shards=self.root/'shards';shards.mkdir()
        rows=[{'id':f'a_gen{i}','primary_id':'a','input_ids':[1,2,3],'loss_mask':[0,1,1],
               'metadata':{'finish_reason':'length','sampling_params':{'temperature':.8}}} for i in range(2)]
        rows.append(dict(rows[0],id='b_gen0',primary_id='b'))
        f=shards/'source_00_out.jsonl';f.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        result=verify(shards,expected_primary_ids={'a','b'},expected_sampling={'temperature':.8})
        self.assertEqual((result['rows'],result['conversations']),(3,2))
        with self.assertRaisesRegex(ValueError,'coverage'):verify(shards,expected_primary_ids={'a','b','c'})
        with self.assertRaisesRegex(ValueError,'sampling'):verify(shards,expected_sampling={'temperature':1.})


if __name__=='__main__':unittest.main()
