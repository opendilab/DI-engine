import pytest
import numpy as np
import torch
from nervex.data import timestep_collate

B, T = 4, 3


@pytest.mark.unittest
class TestTimestepCollate:

    def get_data(self):
        data = {
            'obs': [torch.randn(4) for _ in range(T)],
            'reward': [torch.FloatTensor([0]) for _ in range(T)],
            'done': [False for _ in range(T)],
            'prev_state': [(torch.randn(3), torch.randn(3)) for _ in range(T)],
            'action': [[torch.randn(3), torch.randn(5)] for _ in range(T)],
        }
        return data

    def test(self):
        batch = timestep_collate([self.get_data() for _ in range(B)])
        assert isinstance(batch, dict)
        assert set(batch.keys()) == set(['obs', 'reward', 'done', 'prev_state', 'action'])
        assert batch['obs'].shape == (T, B, 4)
        assert batch['reward'].shape == (T, B)
        assert batch['done'].shape == (T, B) and batch['done'].dtype == torch.bool
        assert isinstance(batch['prev_state'], list)
        assert len(batch['prev_state']) == T and len(batch['prev_state'][0]) == B
        assert isinstance(batch['action'], list) and len(batch['action']) == T
        assert batch['action'][0][0].shape == (B, 3)
        assert batch['action'][0][1].shape == (B, 5)
