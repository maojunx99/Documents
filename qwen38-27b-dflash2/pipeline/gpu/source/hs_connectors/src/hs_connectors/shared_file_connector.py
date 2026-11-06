"""Reuse native vLLM extraction/copy code; only publish disk writes atomically."""
import inspect
import os
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import ExampleHiddenStatesConnector
from safetensors.torch import save_file
from hs_connectors.atomic_file import publish


class AtomicSharedFileConnector(ExampleHiddenStatesConnector):
    def __init__(self, *args, **kwargs):
        if len(inspect.signature(ExampleHiddenStatesConnector._write_tensors).parameters)!=4:
            raise RuntimeError('Unsupported native hidden-state writer API; validate this vLLM version first')
        super().__init__(*args, **kwargs)
        if self.use_lock:
            raise ValueError('shared_file requires use_synchronization_lock=False')

    @staticmethod
    def _write_tensors(tensors, event, filename, lock_fd):
        try:
            event.synchronize()
            publish(filename, lambda path:save_file(tensors,path))
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
