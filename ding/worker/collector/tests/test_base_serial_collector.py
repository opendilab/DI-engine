import pytest
import numpy as np
import torch
from ding.worker.collector.base_serial_collector import to_tensor_transitions


def get_transition():
    return {
        'obs': np.random.random((2, 3)),
        'action': np.random.randint(0, 6, size=(1, )),
        'reward': np.random.random((1, )),
        'done': False,
        'next_obs': np.random.random((2, 3)),
    }


@pytest.mark.unittest
def test_to_tensor_transitions():
    # test case when shallow copy is True
    transition_list = [get_transition() for _ in range(4)]
    tensor_list = to_tensor_transitions(transition_list, shallow_copy_next_obs=True)
    for i in range(len(tensor_list)):
        tensor = tensor_list[i]
        assert isinstance(tensor['obs'], torch.Tensor)
        assert isinstance(tensor['action'], torch.Tensor), type(tensor['action'])
        assert isinstance(tensor['reward'], torch.Tensor)
        assert isinstance(tensor['done'], bool)
        assert 'next_obs' in tensor
        if i < len(tensor_list) - 1:
            assert id(tensor['next_obs']) == id(tensor_list[i + 1]['obs'])
    # test case when shallow copy is False
    transition_list = [get_transition() for _ in range(4)]
    tensor_list = to_tensor_transitions(transition_list, shallow_copy_next_obs=False)
    for i in range(len(tensor_list)):
        tensor = tensor_list[i]
        assert isinstance(tensor['obs'], torch.Tensor)
        assert isinstance(tensor['action'], torch.Tensor)
        assert isinstance(tensor['reward'], torch.Tensor)
        assert isinstance(tensor['done'], bool)
        assert 'next_obs' in tensor
        if i < len(tensor_list) - 1:
            assert id(tensor['next_obs']) != id(tensor_list[i + 1]['obs'])
