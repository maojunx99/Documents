"""Native backend for atomically published shared feature files."""
from pathlib import Path
from safetensors.torch import load_file
from hs_connectors.transfer import FileBackend, FileTransfer, HiddenStatesBackend
from hs_connectors.atomic_file import wait


class SharedFileTransfer(FileTransfer):
    def get_cached(self, file_idx):
        path = self.hidden_states_path/f'hs_{file_idx}.safetensors'
        return load_file(path) if path.exists() else None

    def get_generated(self, handle):
        path = Path(handle)
        if path.is_symlink() or path.parent.resolve()!=self.hidden_states_path.resolve() or path.suffix!='.safetensors':
            raise ValueError('Feature handle is outside the shared job cache')
        return load_file(wait(path))


@HiddenStatesBackend.register('shared_file')
class SharedFileBackend(FileBackend):
    # FileBackend already declares --hidden-states-path in the common native parser.
    @staticmethod
    def add_launch_args(parser):
        pass

    @staticmethod
    def add_train_args(parser):
        pass

    @staticmethod
    def from_train_args(args, data_path):
        return SharedFileTransfer(Path(args.hidden_states_path) if args.hidden_states_path else Path(data_path)/'hidden_states')

    @staticmethod
    def build_kv_transfer_config(args):
        return {'kv_connector':'AtomicSharedFileConnector','kv_role':'kv_producer',
            'kv_connector_module_path':'hs_connectors.shared_file_connector',
            'kv_connector_extra_config':{'shared_storage_path':args.hidden_states_path,
                                        'use_synchronization_lock':False}}
