import torch
import torch.nn as nn
from ding.torch_utils.optimizer_helper import Adam, RMSprop, calculate_grad_norm, \
    calculate_grad_norm_without_bias_two_norm
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


def try_optim_with(tname, t, optim_t):
    net = LinearNet()
    mse_fn = nn.L1Loss()
    if tname == 'grad_clip':
        if optim_t == 'rmsprop':
            optimizer = RMSprop(
                net.parameters(),
                grad_clip_type=t,
                clip_value=0.000001,
                clip_norm_type=1.2,
                lr=0.1,
                clip_momentum_timestep=2,
                ignore_momentum_timestep=2,
                clip_coef=0.5
            )
        else:
            optimizer = Adam(
                net.parameters(),
                grad_clip_type=t,
                clip_value=0.000001,
                clip_norm_type=1.2,
                lr=0.1,
                optim_type=optim_t,
                clip_momentum_timestep=2,
                ignore_momentum_timestep=2,
                clip_coef=0.5
            )
    if tname == 'grad_ignore':
        if optim_t == 'rmsprop':
            optimizer = RMSprop(
                net.parameters(),
                grad_ignore_type=t,
                clip_value=0.000001,
                ignore_value=0.000001,
                ignore_norm_type=1.2,
                lr=0.1,
                clip_momentum_timestep=2,
                ignore_momentum_timestep=2,
            )
        else:
            optimizer = Adam(
                net.parameters(),
                grad_ignore_type=t,
                clip_value=0.000001,
                ignore_value=0.000001,
                ignore_norm_type=1.2,
                lr=0.1,
                optim_type=optim_t,
                clip_momentum_timestep=2,
                ignore_momentum_timestep=2,
                ignore_coef=0.01
            )
    # 网络输入和标签
    x = torch.FloatTensor([120])
    x.requires_grad = True
    target_value = torch.FloatTensor([2])
    target_value.requires_grad = True
    # loss计算
    for _ in range(10):
        predict = net(x)
        loss = mse_fn(predict, target_value)
        loss.backward()
        optimizer.step()
    if t is not None and 'ignore' not in t:
        assert optimizer.get_grad() != 0.
    for _ in range(10):
        target_value = torch.FloatTensor([_ ** 2])
        target_value.requires_grad = True
        predict = net(x)
        loss = mse_fn(predict, target_value)
        loss.backward()
        optimizer.step()

    if t is None:
        print("weight without optimizer clip:" + str(net.linear.weight))
    else:
        print("weight with optimizer {} of type: {} is ".format(tname, t) + str(net.linear.weight))

    weight = net.linear.weight
    return weight


@pytest.mark.unittest
class TestAdam:

    def test_naive(self):
        support_type = {
            'optim': ['adam', 'adamw'],
            'grad_clip': [None, 'clip_momentum', 'clip_value', 'clip_norm', 'clip_momentum_norm'],
            'grad_norm': [None],
            'grad_ignore': [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', 'ignore_momentum_norm'],
        }

        for optim_t in support_type['optim']:
            for tname in ['grad_clip', 'grad_ignore']:
                for t in support_type[tname]:
                    try_optim_with(tname=tname, t=t, optim_t=optim_t)


@pytest.mark.unittest
class TestRMSprop:

    def test_naive(self):
        support_type = {
            'grad_clip': [None, 'clip_momentum', 'clip_value', 'clip_norm', 'clip_momentum_norm'],
            'grad_norm': [None],
            'grad_ignore': [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', 'ignore_momentum_norm'],
        }

        for tname in ['grad_clip', 'grad_ignore']:
            for t in support_type[tname]:
                try_optim_with(tname=tname, t=t, optim_t='rmsprop')


@pytest.mark.unittest
class Test_calculate_grad_norm_with_without_bias:

    def test_two_functions(self):
        net = LinearNet()
        mse_fn = nn.L1Loss()
        optimizer = Adam(net.parameters(), )
        x = torch.FloatTensor([120])
        x.requires_grad = True
        target_value = torch.FloatTensor([2])
        target_value.requires_grad = True
        for _ in range(10):
            predict = net(x)
            loss = mse_fn(predict, target_value)
            loss.backward()
            optimizer.step()
        inf_norm = calculate_grad_norm(model=net, norm_type='inf')
        two_norm = calculate_grad_norm(model=net)
        two_norm_nobias = float(calculate_grad_norm_without_bias_two_norm(model=net))
        one_norm = calculate_grad_norm(model=net, norm_type=1)
        assert isinstance(two_norm, float)
        assert isinstance(inf_norm, float)
        assert isinstance(one_norm, float)
        assert isinstance(two_norm_nobias, float)
