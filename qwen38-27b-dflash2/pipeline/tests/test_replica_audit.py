"""CPU contract tests for failure classification and replica/shard mapping."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'npu/scripts'))
from replica_audit import compare_replicas


class ReplicaAuditTests(unittest.TestCase):
    def payloads(self):
        # Different shards are intentionally different. Corresponding replicas match.
        return [{'tensors': {'model/w': {'sha256': str(rank % 4)},
                            'optimizer0/w/exp_avg': {'sha256': str(rank % 4)}},
                 'optimizer_iteration_order': [['w', 'b']]} for rank in range(8)]

    def test_corresponding_shards_only(self):
        self.assertEqual(compare_replicas(self.payloads(), 2, 4), [])

    def test_parameter_difference_names_the_pair_and_tensor(self):
        states = self.payloads()
        states[5]['tensors']['model/w']['sha256'] = 'different'
        self.assertEqual(compare_replicas(states, 2, 4),
                         [{'ranks': [1, 5], 'keys': ['model/w']}])

    def test_optimizer_difference_is_distinguished(self):
        states = self.payloads()
        states[6]['tensors']['optimizer0/w/exp_avg']['sha256'] = 'different'
        self.assertEqual(compare_replicas(states, 2, 4),
                         [{'ranks': [2, 6], 'keys': ['optimizer0/w/exp_avg']}])

    def test_iteration_order_does_not_create_a_false_difference(self):
        states = self.payloads()
        states[4]['optimizer_iteration_order'][0].reverse()
        self.assertEqual(compare_replicas(states, 2, 4), [])

    def test_missing_state_is_a_failure(self):
        states = self.payloads()
        del states[7]['tensors']['optimizer0/w/exp_avg']
        self.assertEqual(compare_replicas(states, 2, 4),
                         [{'ranks': [3, 7], 'keys': ['optimizer0/w/exp_avg']}])

    def test_three_replicas(self):
        states = self.payloads()
        states.extend(copy.deepcopy(states[:4]))
        states[9]['tensors']['model/w']['sha256'] = 'different'
        self.assertEqual(compare_replicas(states, 3, 4),
                         [{'ranks': [1, 9], 'keys': ['model/w']}])

    def test_mesh_size_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_replicas(self.payloads(), 4, 4)


if __name__ == '__main__':
    unittest.main()
