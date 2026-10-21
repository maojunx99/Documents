"""Check model-type routing without importing an accelerator training stack."""
import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace
import unittest
import os

SOURCE=Path(__file__).resolve().parents[1]/'npu/source/src/speculators/models/utils.py'

class TargetNorm(unittest.TestCase):
    def test_actual_target_type_selects_gain_and_not_model_directory_name(self):
        tree=ast.parse(SOURCE.read_text())
        selected=[]
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.Assign)):
                name=node.name if isinstance(node,ast.FunctionDef) else getattr(node.targets[0],'id','')
                if name in {'GEMMA_STYLE_FINAL_NORM_MODEL_TYPES','_verifier_model_type',
                            'uses_gemma_style_final_norm','resolve_verifier_norm_class'}:selected.append(node)
        env={'cache':cache,'Gemma3RMSNorm':object(),'Qwen3RMSNorm':object(),
             'get_verifier_config':lambda name:SimpleNamespace(model_type=name)}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),env)
        for name,gain in [('qwen3_5_text',True),('qwen3_5',True),('gemma3_text',True),
                          ('qwen3',False),('gemma4',False),('gemma3n',False)]:
            c=SimpleNamespace(speculators_config=SimpleNamespace(verifier=SimpleNamespace(name_or_path=name)))
            self.assertEqual(env['uses_gemma_style_final_norm'](c),gain)
            self.assertIs(env['resolve_verifier_norm_class'](c),env['Gemma3RMSNorm'] if gain else env['Qwen3RMSNorm'])
        self.assertIs(env['resolve_verifier_norm_class'](SimpleNamespace()),env['Qwen3RMSNorm'])

    def test_missing_teacher_sample_is_not_silently_removed_by_collation(self):
        path=SOURCE.parents[4]/'scripts/run_train_parallel.py'
        tree=ast.parse(path.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='strict_get_raw_data')
        captured=[];sentinel=object()
        env={'original_get_raw_data':lambda self,index:sentinel,'save':lambda *x:captured.append(x),'os':os,'RANK':3}
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),'exec'),env)
        self.assertIs(env['strict_get_raw_data'](None,5),sentinel)
        env['original_get_raw_data']=lambda self,index:None
        with self.assertRaisesRegex(RuntimeError,'refusing a dropped record'):env['strict_get_raw_data'](None,7)
        self.assertEqual(captured[0][1]['index'],7)

if __name__=='__main__':unittest.main()
