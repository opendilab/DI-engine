from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.agents.utils_torch import CategoricalPd


class PpoPolicyBase(nn.Module):
    def forward(self, inputs, mode=None):
        assert(mode in ['step', 'value'])
        f = getattr(self, mode)
        return f(inputs)

    def step(self, inputs):
        raise NotImplementedError

    def value(self, inputs):
        raise NotImplementedError


class MlpPolicy(PpoPolicyBase):
    def __init__(self, ob_space, ac_space, fc_dim=512):
        super(MlpPolicy, self).__init__()
        if isinstance(ac_space, MaskDiscrete):
            ob_space, mask_space = ob_space.spaces
        self.use_mask = isinstance(ac_space, MaskDiscrete)

        ob_space_flatten_dim = reduce(lambda x, y: x*y, ob_space.shape)
        self.act = nn.Tanh()
        self.pi_h1 = nn.Linear(ob_space_flatten_dim, fc_dim)
        self.pi_h2 = nn.Linear(fc_dim, fc_dim)
        self.pi_h3 = nn.Linear(fc_dim, fc_dim)
        self.vf_h1 = nn.Linear(ob_space_flatten_dim, fc_dim)
        self.vf_h2 = nn.Linear(fc_dim, fc_dim)
        self.vf_h3 = nn.Linear(fc_dim, fc_dim)
        self.vf = nn.Linear(fc_dim, 1)
        self.pi_logit = nn.Linear(fc_dim, ac_space.n)

        self.pd = CategoricalPd()
        self.initial_state = None
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.orthogonal_(self.pi_logit.weight, 0.01)
        torch.nn.init.constant_(self.pi_logit.bias, 0.0)

    def step(self, inputs):
        x = inputs['obs']
        B = x.shape[0]
        x = x.view(B, -1)
        vf = self.act(self.vf_h1(x))
        vf = self.act(self.vf_h2(vf))
        vf = self.act(self.vf_h3(vf))
        vf = self.vf(vf)

        pi = self.act(self.pi_h1(x))
        pi = self.act(self.pi_h2(pi))
        pi = self.act(self.pi_h3(pi))
        pi_logit = self.pi_logit(pi)

        if self.use_mask:
            mask = inputs['mask']
            assert(mask is not None)
            pi_logit -= (1-mask) * 1e30
        self.pd.update_logits(pi_logit)
        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)
        return action, vf, self.initial_state, neglogp

    def value(self, inputs):
        x = inputs['obs']
        B = x.shape[0]
        x = x.view(B, -1)
        vf = self.act(self.vf_h1(x))
        vf = self.act(self.vf_h2(vf))
        vf = self.act(self.vf_h3(vf))
        vf = self.vf(vf)
        return vf


class LSTMFC(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(LSTMFC, self).__init__()
        self.wx = nn.Parameter(torch.zeros(input_dim, hidden_dim*4))
        self.wh = nn.Parameter(torch.zeros(hidden_dim, hidden_dim*4))
        self.bias = nn.Parameter(torch.zeros(hidden_dim*4))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _init(self):
        torch.nn.init.orthogonal_(self.wx, 1.0)
        torch.nn.init.orthogonal_(self.wh, 1.0)
        torch.nn.init.constant_(self.bias, 0.0)

    def batch_to_seq(self, x):
        other_dim = [2 + i for i in range(len(x.shape) - 2)]
        return x.permute(1, 0, *other_dim).contiguous()

    def seq_to_batch(self, x):
        other_dim = [2 + i for i in range(len(x.shape) - 2)]
        return x.permute(1, 0, *other_dim).contiguous()

    def forward(self, inputs, done, state):
        inputs = self.batch_to_seq(inputs)
        done = self.batch_to_seq(done)
        outputs = []
        c, h = torch.chunk(state, 2, dim=1)
        for idx, (x, d) in enumerate(zip(inputs, done)):
            c.mul_(1 - d)
            h.mul_(1 - d)
            z = torch.matmul(x, self.wx) + torch.matmul(h, self.wh) + self.bias
            i, f, o, u = torch.chunk(z, 4, dim=1)
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            o = F.sigmoid(o)
            u = F.tanh(u)
            c = f * c + i * u
            h = o * F.tanh(c)
            outputs.append(h)
        state = torch.cat([c, h], dim=1)
        outputs = torch.stack(outputs, dim=0)
        outputs = self.seq_to_batch(outputs)
        return outputs, state

    def __repr__(self):
        return 'input_dim: {}\thidden_dim: {}'.format(
                self.input_dim, self.hidden_dim)


class LstmPolicy(PpoPolicyBase):
    def __init__(self, ob_space, ac_space, unroll_length,
                 fc_dim=512, lstm_dim=512):
        super(LstmPolicy, self).__init__()
        if isinstance(ac_space, MaskDiscrete):
            ob_space, mask_space = ob_space.spaces
        self.use_mask = isinstance(ac_space, MaskDiscrete)

        ob_space_flatten_dim = reduce(lambda x, y: x*y, ob_space.shape)
        self.relu = nn.ReLU()
        self.fc_h1 = nn.Linear(ob_space_flatten_dim, fc_dim)
        self.fc_h2 = nn.Linear(fc_dim, fc_dim)
        self.vf = nn.Linear(fc_dim, 1)
        self.lstm = LSTMFC(fc_dim, lstm_dim)
        self.pi_logit = nn.Linear(fc_dim, ac_space.n)
        self.pd = CategoricalPd()
        self.initial_state = torch.zeros(lstm_dim*2)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.orthogonal_(self.pi_logit.weight, 0.01)
        torch.nn.init.constant_(self.pi_logit.bias, 0.0)
        self.lstm._init()

    def _get_h(self, x):
        h = self.relu(self.fc_h1(x))
        h = self.relu(self.fc_h2(h))
        return h

    def step(self, inputs):
        x, state, done = inputs['obs'], inputs['state'], inputs['done']
        B, S = x.shape[:2]
        x = x.view(B*S, -1)
        h = self._get_h(x)
        h = h.view(B, S, -1)
        h, snew = self.lstm(h, done, state)
        h = h.view(B*S, -1)
        pi_logit = self.pi_logit(h)
        vf = self.vf(h)[:, 0]

        if self.use_mask:
            mask = inputs['mask']
            assert(mask is not None)
            pi_logit -= (1-mask) * 1e30
        self.pd.update_logits(pi_logit)
        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)
        return action, vf, snew, neglogp

    def value(self, inputs):
        x, state, done = inputs['obs'], inputs['state'], inputs['done']
        B, S = x.shape[:2]
        x = x.view(B*S, -1)
        h = self._get_h(x)
        h = h.view(B, S, -1)
        h, snew = self.lstm(h, done, state)
        vf = self.vf(h)[:, 0]
        return vf


def test_mlp_policy():
    class T():
        pass
    inputs = {}
    inputs['obs'] = torch.randn(4, 3, 32, 32)
    ob_space = torch.empty(3, 32, 32)
    ac_space = T()
    setattr(ac_space, 'n', 10)
    model = MlpPolicy(ob_space, ac_space)
    print(model)
    output_v = model(inputs, mode='value')
    print(output_v.shape)
    output_step = model(inputs, mode='step')
    for item in output_step:
        if item is not None:
            print(item.shape)
        else:
            print('none')


def test_lstm_policy():
    class T():
        pass
    inputs = {}
    inputs['obs'] = torch.randn(4, 5, 3, 32, 32)
    inputs['done'] = torch.ByteTensor(4, 5, 1).float()
    inputs['state'] = torch.randn(4, 512*2)
    ob_space = torch.empty(3, 32, 32)
    ac_space = T()
    setattr(ac_space, 'n', 10)
    model = LstmPolicy(ob_space, ac_space, unroll_length=5)
    print(model)
    output_v = model(inputs, mode='value')
    print('output_v', output_v.shape)
    output_step = model(inputs, mode='step')
    for item in output_step:
        if item is not None:
            print(item.shape)
        else:
            print('none')


if __name__ == "__main__":
    test_mlp_policy()
    test_lstm_policy()
