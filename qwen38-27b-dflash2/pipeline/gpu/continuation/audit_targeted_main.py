"""Finish smoke audit and real CUDA restore only; never dispatch main training."""
import argparse,collections,json,math,os,subprocess,time,traceback
from pathlib import Path
from prepare_inventory import write
from run_additional_generation import ident,alive,owned_tree as original_owned_tree,available_gpus,digest

def owned_tree(rec):
    try:return original_owned_tree(rec)
    except AssertionError:
        # A root can exit between ident() and alive(); only ignore proven exit.
        now=ident(rec["pid"])
        if now is None or now["starttime"]!=rec["starttime"] or now["state"]=="Z":return []
        raise


def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]

def audit_training(root,child):
    spec=json.loads((child/'RUN_SPEC.json').read_text())
    schedule=json.loads(Path(spec['schedule']).read_text());initials=[];times=[];mem=[]
    for arm in ['overlap','sampled_ce']:
        for rank in range(2):
            d=child/arm/'audit';initial=json.loads((d/f'initial_rank{rank}.json').read_text());initials.append(initial['parameter_sha256'])
            assert initial['trainable_tensors']==81 and initial['optimizer_states']==0
            assert initial['master_dtype']=='float32' and initial['model_dtype']=='bfloat16'
            completed=json.loads((d/f'completed_rank{rank}.json').read_text())
            assert completed['completed_optimizer_steps']==697 and completed['next_microbatch']==2788
            assert completed['actual_input_tokens']==schedule['actual_input_tokens']
            assert completed['optimizer_moments_dtype']=='float32' and completed['frozen_parameters_unchanged']
            assert completed['final_parameter_sha256']!=initial['parameter_sha256']
            inp=lines(d/f'inputs_rank{rank}.jsonl');feat=lines(d/f'features_rank{rank}.jsonl');metrics=lines(d/f'metrics_rank{rank}.jsonl')
            assert len(inp)==len(feat)==len(metrics)==2980
            assert collections.Counter(x['phase'] for x in inp)==dict(baseline=96,train=2788,final=96)
            for phase,expected in [('baseline',96),('train',2788),('final',96)]:
                rec=[x for x in inp if x['phase']==phase];assert [x['step'] for x in rec]==list(range(expected))
                for i,(a,b,c) in enumerate(zip([x for x in inp if x['phase']==phase],
                        [x for x in feat if x['samples'][0]['phase']==phase],
                        [x for x in metrics if x['phase']==phase],strict=True)):
                    pack=schedule['train_packs' if phase=='train' else 'val_packs'][str(rank)][i]
                    assert [x['index'] for x in b['samples']]==pack
                    assert a['input_sha256']==b['input_sha256'] and a['documents_sha256']==b['documents_sha256']
                    assert a['input_tokens']==b['nonpadding_input_tokens']
                    assert c['valid_positions']>0 and math.isfinite(c['loss'])
                    assert all(math.isfinite(v) for v in c['metrics'].values())
            baseline=[{k:v for k,v in x.items() if k!='phase'} for x in inp if x['phase']=='baseline']
            final=[{k:v for k,v in x.items() if k!='phase'} for x in inp if x['phase']=='final'];assert baseline==final
            progress=json.loads((d/f'progress_rank{rank}.json').read_text());times.append(progress['elapsed_s']);mem.append(progress['peak_memory_bytes'])
    assert all(h==initials[0] for h in initials)
    for rank in range(2):
        for name in ['inputs','features']:
            assert lines(child/'overlap/audit'/f'{name}_rank{rank}.jsonl')==lines(child/'sampled_ce/audit'/f'{name}_rank{rank}.jsonl')
    router=lines(child/'router_routing.jsonl')
    # Router entries can include startup model-list requests; all completed HTTP
    # responses are required to succeed, never hidden-state skip-on-error.
    assert router and len({r['sequence'] for r in router})==len(router)
    for r in router:assert r['http_status']==200,r
    for p,h in spec['fingerprinted_files'].items():assert digest(Path(p))==h,p
    after=json.loads((child/'fingerprints_after.json').read_text());assert after==spec['fingerprinted_files']
    full=json.loads(Path(spec['selection_lock']).with_name('native_schedule.json').read_text())
    projected=max(times)
    assert projected<28800,'Smoke projects above fixed main training8h limit; no main launch'
    result=dict(passed=True,time=time.time(),four_ranks_steps=697,records_per_rank=2980,initial_tensors=81,
        all_inputs_anchors_hidden_paired=True,baseline_final_input_anchors_paired=True,finite_all_losses=True,
        actual_input_tokens_per_arm=schedule['actual_input_tokens'],max_elapsed_697updates_s=max(times),
        max_peak_memory_GiB=max(mem)/1024**3,observed_training_seconds=projected,
        limits='Observed training timing excludes initialization and final validation',
        source_target_initial_unchanged=True,main_training_started=True)
    write(child/'independent_update_audit.json',result);return spec,result
