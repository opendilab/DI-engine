import pytest
import torch
import torch.nn as nn

from ding.torch_utils.nn_test_helper import is_differentiable


@pytest.mark.unittest
def test_is_differentibale():

    class LinearNet(nn.Module):

        def __init__(self, features_in=1, features_out=1):
            super().__init__()
            self.linear = nn.Linear(features_in, features_out)
            self._init_weight()

        def forward(self, x):
            return self.linear(x)

        def _init_weight(self):
            nn.init.constant_(self.linear.weight, val=1)
            nn.init.constant_(self.linear.bias, val=0)

    net = LinearNet()
    mse_fn = nn.L1Loss()
    net._init_weight()
    x = torch.FloatTensor([120])
    target_value = torch.FloatTensor([2])
    target_value.requires_grad = True
    loss = mse_fn(net(x), target_value)
    assert is_differentiable(loss, net) is None
    with pytest.raises(AssertionError):
        value = net(x).detach()
        target_value = torch.FloatTensor([2])
        target_value.requires_grad = False
        is_differentiable(loss, net)
