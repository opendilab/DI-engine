import unittest
import pytest
import numpy as np
import torch
import treetensor.torch as ttorch

from ding.policy.common_utils import default_preprocess_learn

shape_test = [
    [2],
    [1],
]

dtype_test = [
    "int32",
    "int64",
    "float32",
    "float64",
]

data_type_test = [
    "numpy",
    "torch",
    "treetensor",
]


def get_action(shape, dtype, class_type):
    if class_type == "numpy":
        return np.random.randn(*shape).astype(dtype)
    else:
        if dtype == "int32":
            dtype = torch.int32
        elif dtype == "int64":
            dtype = torch.int64
        elif dtype == "float16":
            dtype = torch.float16
        elif dtype == "float32":
            dtype = torch.float32
        elif dtype == "float64":
            dtype = torch.float64

        if class_type == "torch":
            return torch.randn(*shape).type(dtype)
        elif class_type == "treetensor":
            return ttorch.randn(*shape).type(dtype)


@pytest.mark.unittest
def test_default_preprocess_learn_action():

    for shape in shape_test:
        for dtype in dtype_test:
            for data_type in data_type_test:

                data = [
                    {
                        'obs': np.random.randn(4, 84, 84),
                        'action': get_action(shape, dtype, data_type),
                        'reward': 1.0,
                        'next_obs': np.random.randn(4, 84, 84),
                        'done': False,
                        'weight': 1.0,
                        'value': 1.0,
                        'adv': 1.0,
                    } for _ in range(10)
                ]
                use_priority_IS_weight = False
                use_priority = False
                use_nstep = False
                ignore_done = False
                data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

                assert data['obs'].shape == torch.Size([10, 4, 84, 84])
                if dtype in ["int32", "int64"] and shape[0] == 1:
                    assert data['action'].shape == torch.Size([10])
                else:
                    assert data['action'].shape == torch.Size([10, *shape])
                assert data['reward'].shape == torch.Size([10])
                assert data['next_obs'].shape == torch.Size([10, 4, 84, 84])
                assert data['done'].shape == torch.Size([10])
                assert data['weight'].shape == torch.Size([10])
                assert data['value'].shape == torch.Size([10])
                assert data['adv'].shape == torch.Size([10])


@pytest.mark.unittest
def test_default_preprocess_learn_reward_done_adv_1d():

    data = [
        {
            'obs': np.random.randn(4, 84, 84),
            'action': np.random.randn(2),
            'reward': np.array([1.0]),
            'next_obs': np.random.randn(4, 84, 84),
            'done': False,
            'value': np.array([1.0]),
            'adv': np.array([1.0]),
        } for _ in range(10)
    ]
    use_priority_IS_weight = False
    use_priority = False
    use_nstep = False
    ignore_done = False
    data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

    assert data['reward'].shape == torch.Size([10])
    assert data['done'].shape == torch.Size([10])
    assert data['weight'] is None
    assert data['value'].shape == torch.Size([10])
    assert data['adv'].shape == torch.Size([10])


@pytest.mark.unittest
def test_default_preprocess_learn_ignore_done():
    data = [
        {
            'obs': np.random.randn(4, 84, 84),
            'action': np.random.randn(2),
            'reward': np.array([1.0]),
            'next_obs': np.random.randn(4, 84, 84),
            'done': True,
            'value': np.array([1.0]),
            'adv': np.array([1.0]),
        } for _ in range(10)
    ]
    use_priority_IS_weight = False
    use_priority = False
    use_nstep = False
    ignore_done = True
    data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

    assert data['done'].dtype == torch.float32
    assert torch.sum(data['done']) == 0


@pytest.mark.unittest
def test_default_preprocess_learn_use_priority_IS_weight():
    data = [
        {
            'obs': np.random.randn(4, 84, 84),
            'action': np.random.randn(2),
            'reward': 1.0,
            'next_obs': np.random.randn(4, 84, 84),
            'done': False,
            'priority_IS': 1.0,
            'value': 1.0,
            'adv': 1.0,
        } for _ in range(10)
    ]
    use_priority_IS_weight = True
    use_priority = True
    use_nstep = False
    ignore_done = False
    data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

    assert data['weight'].shape == torch.Size([10])
    assert torch.sum(data['weight']) == torch.tensor(10.0)


@pytest.mark.unittest
def test_default_preprocess_learn_nstep():
    data = [
        {
            'obs': np.random.randn(4, 84, 84),
            'action': np.random.randn(2),
            'reward': np.array([1.0, 2.0, 0.0]),
            'next_obs': np.random.randn(4, 84, 84),
            'done': False,
            'value': 1.0,
            'adv': 1.0,
        } for _ in range(10)
    ]
    use_priority_IS_weight = False
    use_priority = False
    use_nstep = True
    ignore_done = False
    data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

    assert data['reward'].shape == torch.Size([3, 10])
    assert data['reward'][0][0] == torch.tensor(1.0)
    assert data['reward'][1][0] == torch.tensor(2.0)
    assert data['reward'][2][0] == torch.tensor(0.0)
