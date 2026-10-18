"""Occurrence-keyed native features and strict 8192-token packing.

No service launch, model load, sampler reselection, or skip-on-error path.
Actual FileTransfer roundtrip still requires the separate GPU smoke.
"""
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset,Sampler
from bounded_feature_cache import BoundedSharedFeatureCache
from shared_feature_cache import tensor_sha


def occurrence_seed(seed,phase,rank,microbatch,index):
    # Baseline and final validation intentionally share the same RNG schedule.
    phase='validation' if phase in ('baseline','final') else phase
    value=f'{seed}/{phase}/{rank}/{microbatch}/{index}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'big')%(2**63-1)


class NativeHiddenProducer:
    def __init__(self,endpoint,model,transport_root,timeout=600):
        self.endpoint=endpoint;self.model=model;self.transport_root=Path(transport_root)
        self.timeout=timeout;self.client=None;self.transfer=None

    def __call__(self,input_ids):
        import openai
        from hs_connectors import FileTransfer
        from speculators.data_generation.vllm_client import generate_hidden_states
        from speculators.data_generation.offline import check_hidden_states
        if self.client is None:
            self.client=openai.OpenAI(base_url=self.endpoint,api_key='EMPTY',max_retries=0)
            if [m.id for m in self.client.models.list().data]!=[self.model]:
                raise ValueError('Teacher model identity mismatch')
            self.transfer=FileTransfer(self.transport_root);self.transfer.setup()
        handle=None
        try:
            # Call the exact native body once: the installed wrapper has no max_retries keyword.
            import inspect
            single_request=inspect.unwrap(generate_hidden_states)
            assert single_request is not generate_hidden_states, 'Expected native retry wrapper'
            inspect.signature(single_request).bind(self.client,self.model,{'input_ids':input_ids.tolist()},timeout=self.timeout)
            handle=single_request(self.client,self.model,{'input_ids':input_ids.tolist()},timeout=self.timeout)
            tensors=self.transfer.get_generated(handle)
            if tensors is None:raise RuntimeError('Missing native file transport payload')
            check_hidden_states(tensors,input_ids.tolist())
            # Native file mappings never survive transport deletion.
            result={k:tensors[k].detach().cpu().clone() for k in ('hidden_states','token_ids')}
            self.transfer.delete(handle);handle=None
            return result
        finally:
            if handle is not None:self.transfer.delete(handle)


class ScheduledSharedDataset(Dataset):
    def __init__(self,records,*,cache_root,arm,producer,seed,hidden_size=5120,tapped_layers=5,noise_std=.05):
        self.records=records;self.cache=BoundedSharedFeatureCache(cache_root,arm)
        self.producer=producer;self.seed=seed;self.hidden_size=hidden_size
        self.tapped_layers=tapped_layers;self.noise_std=noise_std

    def __len__(self):return len(self.records)

    def __getitem__(self,key):
        phase,rank,micro,index=key
        if phase not in ('train','baseline','final','validation'):raise ValueError('Unknown phase')
        raw=self.records[index]
        ids=torch.as_tensor(raw['input_ids'],dtype=torch.long).clone()
        supplied=torch.as_tensor(raw['loss_mask'])
        if ids.ndim!=1 or supplied.shape!=ids.shape or not 1<len(ids)<=4096:raise ValueError('Invalid prepared sequence shape/length')
        if not torch.all((supplied==0)|(supplied==1)).item():raise ValueError('Nonbinary loss mask')
        mask=supplied.bool().clone()
        if not mask.any().item():raise ValueError('Empty supervision')
        occurrence=f'{phase}/rank{rank}/micro{micro}/row{index}'
        tensors,meta=self.cache.fetch(occurrence,ids,lambda:self.producer(ids),hidden_shape=(len(ids),self.tapped_layers+1,self.hidden_size))
        h=tensors['hidden_states']
        raw_hidden=h[:,:-1].flatten(1);hidden=raw_hidden
        noise_seed=occurrence_seed(self.seed,phase,rank,micro,index)
        if phase=='train':
            gen=torch.Generator(device='cpu');gen.manual_seed(noise_seed)
            # Same operation order/dtype as native AddUniformNoise.
            hidden=raw_hidden+2*(torch.rand(raw_hidden.shape,dtype=raw_hidden.dtype,generator=gen)-.5)*self.noise_std
        result=dict(input_ids=ids,loss_mask=mask,hidden_states=hidden,
                    verifier_last_hidden_states=h[:,-1],lengths=torch.tensor([len(ids)]),position_ids=torch.arange(len(ids)))
        result['_audit']=dict(occurrence=occurrence,phase=phase,rank=rank,microbatch=micro,index=index,
            teacher_sha256=meta['sha256'],input_sha256=tensor_sha(ids),mask_sha256=tensor_sha(mask),
            raw_hidden_sha256=tensor_sha(raw_hidden),noisy_hidden_sha256=tensor_sha(hidden),
            last_hidden_sha256=tensor_sha(h[:,-1]),noise_seed=noise_seed)
        return result


class LockedBatchSampler(Sampler):
    """Consume precomputed row-index packs, without shuffling or truncation."""
    def __init__(self,packs,*,phase,rank,start_microbatch=0):
        self.packs=packs;self.phase=phase;self.rank=rank;self.start=start_microbatch
        if not 0<=self.start<=len(packs):raise ValueError('Invalid sealed schedule cursor')

    def __len__(self):return len(self.packs)-self.start

    def __iter__(self):
        for micro in range(self.start,len(self.packs)):
            rows=self.packs[micro]
            if not rows or len(set(rows))!=len(rows):raise ValueError('Empty/duplicate row in frozen pack')
            yield [(self.phase,self.rank,micro,index) for index in rows]


class StrictNativeCollate:
    def __init__(self,*,hidden_size=5120,tapped_layers=5,packed_budget=8192):
        from speculators.train.data import CollateFn
        self.budget=packed_budget
        self.native=CollateFn(max_len=packed_budget,hidden_size=hidden_size,num_target_layers=tapped_layers,dtype=torch.bfloat16)

    def __call__(self,batch):
        if not batch or any(not isinstance(row,dict) for row in batch):raise ValueError('Missing data cannot be dropped')
        audits=[row['_audit'] for row in batch]
        if len({(a['phase'],a['rank'],a['microbatch']) for a in audits})!=1:raise ValueError('Mixed frozen microbatches')
        lengths=[int(row['input_ids'].numel()) for row in batch];total=sum(lengths)
        if total>self.budget:raise ValueError('Frozen pack overflows; native silent truncation forbidden')
        clean=[{k:v for k,v in row.items() if k!='_audit'} for row in batch]
        packed=self.native(clean)
        if packed.pop('error_records')!=0:raise ValueError('Native collator dropped records')
        expected_ids=torch.cat([row['input_ids'] for row in batch])
        expected_mask=torch.cat([row['loss_mask'] for row in batch])
        if not torch.equal(packed['input_ids'][0,:total],expected_ids) or not torch.equal(packed['loss_mask'][0,:total],expected_mask):raise ValueError('Packed token/mask changed')
        offset=0
        for doc,(length,row) in enumerate(zip(lengths,batch,strict=True)):
            if not torch.all(packed['document_ids'][0,offset:offset+length]==doc).item():raise ValueError('Document boundary mismatch')
            if not torch.equal(packed['position_ids'][0,offset:offset+length],torch.arange(length)):raise ValueError('Position reset mismatch')
            offset+=length
        if not torch.all(packed['document_ids'][0,total:]==-1).item() or packed['loss_mask'][0,total:].any().item():raise ValueError('Padding can receive supervision')
        packed['_audit']=dict(samples=audits,nonpadding_input_tokens=total,
            input_sha256=tensor_sha(packed['input_ids']),mask_sha256=tensor_sha(packed['loss_mask']),
            documents_sha256=tensor_sha(packed['document_ids']),hidden_sha256=tensor_sha(packed['hidden_states']),
            last_hidden_sha256=tensor_sha(packed['verifier_last_hidden_states']))
        return packed
