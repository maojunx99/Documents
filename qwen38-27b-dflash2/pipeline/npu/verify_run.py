"""Independent run-record checks for configurable replica/shard counts and epoch budgets."""
import json
import math


def verify_training(run,a,resolved):
    audit=run/'audit';world=a.replicas*a.shard_size
    def read(name,rank):return json.loads((audit/f'{name}_rank{rank}.json').read_text())
    assert not list(audit.glob('failure_rank*.json'))
    initial=[read('initial',i) for i in range(world)]
    final=[read('final',i) for i in range(world)]
    assert all(v['passed'] for v in final)
    assert all(v['trainable']==initial[0]['trainable'] and v['optimizer_initial_states']==[0] for v in initial)
    global_count=initial[0]['trainable_parameters_global']
    assert all(v['replicas']==a.replicas and v['shard_size']==a.shard_size and v['sampler_dp_size']==world for v in initial)
    for group in resolved['mesh']:
        assert sum(initial[i]['trainable_parameters_local'] for i in group)==global_count
        assert all(initial[i]['shard_group']==group for i in group)
    for column in range(a.shard_size):
        group=list(range(column,world,a.shard_size))
        assert all(initial[i]['replica_group']==group for i in group)
        states=[read('replica_state',i) for i in group]
        assert all(s['hashes']==states[0]['hashes'] and s['replica_ranks']==group for s in states)
    assert sum(v['trainable_parameters_local'] for v in initial)==a.replicas*global_count
    steps=final[0]['completed_optimizer_steps'];epochs=final[0]['completed_epochs']
    assert steps>0 and all(v['completed_optimizer_steps']==steps and v['completed_epochs']==epochs for v in final)
    assert a.max_steps is None or steps<=a.max_steps
    expected_epochs=a.epochs if a.stop_after_epoch is None else a.stop_after_epoch+1
    assert epochs==list(range(expected_epochs)) or (a.max_steps is not None and steps==a.max_steps)
    maximum=[]
    for rank in range(world):
        updates=[json.loads(l) for l in (audit/f'updates_rank{rank}.jsonl').read_text().splitlines()]
        assert [v['completed_optimizer_steps'] for v in updates]==list(range(1,steps+1))
        assert all(math.isfinite(v['loss']) and v['optimizer_step_min']==v['optimizer_step_max']==v['completed_optimizer_steps'] for v in updates)
        assert read('exit',rank)['exit_code']==0
        assert read('replica_state',rank)['passed']
        rows=[json.loads(l) for l in (audit/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
        assert sum(v['phase']=='train' for v in rows)==steps
        for epoch in epochs:
            val=read(f'validation_epoch{epoch}',rank)
            assert val['metrics']==read(f'validation_epoch{epoch}',0)['metrics']
            assert all(math.isfinite(v) for v in val['metrics'].values())
            assert sum(v['phase']=='validation' and v['epoch']==epoch for v in rows)==val['batches']
        maximum.append({'rank':rank,'allocated_GiB':max(v['npu_peak_allocated_bytes'] for v in updates)/2**30,
                        'reserved_GiB':max(v['npu_peak_reserved_bytes'] for v in updates)/2**30})
    reload=read('checkpoint_reload',0);assert reload['passed']
    reloaded=[read('reloaded_hashes',i) for i in range(world)]
    assert all(v==reloaded[0] for v in reloaded)
    return {'replicas':a.replicas,'shard_size':a.shard_size,'rank_mesh':resolved['mesh'],'steps_per_rank':[steps]*world,
            'completed_epochs':epochs,'trainable_parameters':global_count,'trainable_tensors':len(initial[0]['trainable']),
            'initial_parameters_identical_empty_optimizer':True,'replica_parameter_optimizer_checks':True,
            'checkpoint_save_reload':reload,'memory_peak':maximum,'native_validation':read(f'validation_epoch{epochs[-1]}',0),
            'scope':'Training pipeline checks; not serving acceptance-length evaluation.'}
