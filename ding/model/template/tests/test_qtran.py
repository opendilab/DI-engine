import pytest
from itertools import product
import torch
from ding.model.template import QTran
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
def test_qtran():
    B = 1
    obs_shape = (1, 64, 64)
    act_shape = 2
    # inputs = {
    #     'obs': {'agent_state': torch.randn(B, *obs_shape),
    #             'global_state': torch.randn(B, *obs_shape)},
    #     'prev_state': [[torch.randn(1, 1, *obs_shape) for __ in range(1)] for _ in range(1)],
    #     'action': torch.randn(B, act_shape)
    # }
    model = QTran(1, obs_shape, 4 * 64 * 64, act_shape, [8, 8, 8], 5)
    # model.forward(inputs)
