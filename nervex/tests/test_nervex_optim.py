import torch
import torch.nn as nn
from nervex.torch_utils.optimizer_util import NervexOptim
import pytest
import time


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


@pytest.mark.unittest
class TestNervexOptim:
    def test_naive(self):
        
        net = LinearNet()
        mse_fn = nn.L1Loss()
        optimizer = NervexOptim(net.parameters(), grad_clip_type='clip_const', clip_value=0.000001, lr=0.1)
        # 网络输入和标签
        x = torch.FloatTensor([120])
        x.requires_grad = True
        target_value = torch.FloatTensor([2])
        target_value.requires_grad = True
        # loss计算
        predict = net(x)
        loss = mse_fn(predict, target_value)
        loss.backward()
        optimizer.step()
        print("weight with optimizer clip:" + str(net.linear.weight))
        with_clip_weight = net.linear.weight

        net = LinearNet()
        mse_fn = nn.L1Loss()
        optimizer = NervexOptim(net.parameters(), grad_clip_type=None, clip_value=0.000001, lr=0.1)
        # 网络输入和标签
        x = torch.FloatTensor([120])
        x.requires_grad = True
        target_value = torch.FloatTensor([2])
        target_value.requires_grad = True
        # loss计算
        predict = net(x)
        loss = mse_fn(predict, target_value)
        loss.backward()
        optimizer.step()
        print("weight without optimizer clip:" + str(net.linear.weight))
        without_clip_weight = net.linear.weight

        assert not with_clip_weight == without_clip_weight