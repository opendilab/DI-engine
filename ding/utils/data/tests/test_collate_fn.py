import pytest
from collections import namedtuple
import random
import numpy as np
import torch
from ding.utils.data import timestep_collate, default_collate, default_decollate, diff_shape_collate

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

    def get_multi_shape_state_data(self):
        data = {
            'obs': [torch.randn(4) for _ in range(T)],
            'reward': [torch.FloatTensor([0]) for _ in range(T)],
            'done': [False for _ in range(T)],
            'prev_state': [
                [(torch.randn(3), torch.randn(5)), (torch.randn(4), ), (torch.randn(5), torch.randn(6))]
                for _ in range(T)
            ],
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

        # hidden_state might contain multi prev_states with different shapes
        batch = timestep_collate([self.get_multi_shape_state_data() for _ in range(B)])
        assert isinstance(batch, dict)
        assert set(batch.keys()) == set(['obs', 'reward', 'done', 'prev_state', 'action'])
        assert batch['obs'].shape == (T, B, 4)
        assert batch['reward'].shape == (T, B)
        assert batch['done'].shape == (T, B) and batch['done'].dtype == torch.bool
        assert isinstance(batch['prev_state'], list)
        print(batch['prev_state'][0][0])
        assert len(batch['prev_state']) == T and len(batch['prev_state'][0]
                                                     ) == B and len(batch['prev_state'][0][0]) == 3
        assert isinstance(batch['action'], list) and len(batch['action']) == T
        assert batch['action'][0][0].shape == (B, 3)
        assert batch['action'][0][1].shape == (B, 5)


@pytest.mark.unittest
class TestDefaultCollate:

    def test_numpy(self):
        data = [np.random.randn(4, 3).astype(np.float64) for _ in range(5)]
        data = default_collate(data)
        assert data.shape == (5, 4, 3)
        assert data.dtype == torch.float64
        data = [float(np.random.randn(1)[0]) for _ in range(6)]
        data = default_collate(data)
        assert data.shape == (6, )
        assert data.dtype == torch.float32
        with pytest.raises(TypeError):
            default_collate([np.array(['str']) for _ in range(3)])

    def test_basic(self):
        data = [random.random() for _ in range(3)]
        data = default_collate(data)
        assert data.shape == (3, )
        assert data.dtype == torch.float32
        data = [random.randint(0, 10) for _ in range(3)]
        data = default_collate(data)
        assert data.shape == (3, )
        assert data.dtype == torch.int64
        data = ['str' for _ in range(4)]
        data = default_collate(data)
        assert len(data) == 4
        assert all([s == 'str' for s in data])
        T = namedtuple('T', ['x', 'y'])
        data = [T(1, 2) for _ in range(4)]
        data = default_collate(data)
        assert isinstance(data, T)
        assert data.x.shape == (4, ) and data.x.eq(1).sum() == 4
        assert data.y.shape == (4, ) and data.y.eq(2).sum() == 4
        with pytest.raises(TypeError):
            default_collate([object() for _ in range(4)])

        data = [{'collate_ignore_data': random.random()} for _ in range(4)]
        data = default_collate(data)
        assert isinstance(data, dict)
        assert len(data['collate_ignore_data']) == 4


@pytest.mark.unittest
class TestDefaultDecollate:

    def test(self):
        with pytest.raises(TypeError):
            default_decollate([object() for _ in range(4)])
        data = torch.randn(4, 3, 5)
        data = default_decollate(data)
        print([d.shape for d in data])
        assert len(data) == 4 and all([d.shape == (3, 5) for d in data])
        data = [torch.randn(8, 2, 4), torch.randn(8, 5)]
        data = default_decollate(data)
        assert len(data) == 8 and all([d[0].shape == (2, 4) and d[1].shape == (5, ) for d in data])
        data = {
            'logit': torch.randn(4, 13),
            'action': torch.randint(0, 13, size=(4, )),
            'prev_state': [(torch.zeros(3, 1, 12), torch.zeros(3, 1, 12)) for _ in range(4)],
        }
        data = default_decollate(data)
        assert len(data) == 4 and isinstance(data, list)
        assert all([d['logit'].shape == (13, ) for d in data])
        assert all([d['action'].shape == (1, ) for d in data])
        assert all([len(d['prev_state']) == 2 and d['prev_state'][0].shape == (3, 1, 12) for d in data])


@pytest.mark.unittest
class TestDiffShapeCollate:

    def test(self):
        with pytest.raises(TypeError):
            diff_shape_collate([object() for _ in range(4)])
        data = [
            {
                'item1': torch.randn(4),
                'item2': None,
                'item3': torch.randn(3),
                'item4': np.random.randn(5, 6)
            },
            {
                'item1': torch.randn(5),
                'item2': torch.randn(6),
                'item3': torch.randn(3),
                'item4': np.random.randn(5, 6)
            },
        ]
        data = diff_shape_collate(data)
        assert isinstance(data['item1'], list) and len(data['item1']) == 2
        assert isinstance(data['item2'], list) and len(data['item2']) == 2 and data['item2'][0] is None
        assert data['item3'].shape == (2, 3)
        assert data['item4'].shape == (2, 5, 6)
        data = [
            {
                'item1': 1,
                'item2': 3,
                'item3': 2.0
            },
            {
                'item1': None,
                'item2': 4,
                'item3': 2.0
            },
        ]
        data = diff_shape_collate(data)
        assert isinstance(data['item1'], list) and len(data['item1']) == 2 and data['item1'][1] is None
        assert data['item2'].shape == (2, ) and data['item2'].dtype == torch.int64
        assert data['item3'].shape == (2, ) and data['item3'].dtype == torch.float32
