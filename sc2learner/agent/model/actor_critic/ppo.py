from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.torch_utils import CategoricalPd, CategoricalPdPytorch
from .actor_critic import ActorCriticBase


class PPOMLP(ActorCriticBase):
    def __init__(self, ob_space, ac_space, seed=0, fc_dim=512, action_type='rand', viz=False):
        super(PPOMLP, self).__init__()
        assert(action_type in ['rand', 'fixed', 'sample'])
        self.action_type = action_type
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

        self.pd = CategoricalPd
        self.initial_state = None
        self.viz = viz
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.orthogonal_(self.pi_logit.weight, 0.01)
        torch.nn.init.constant_(self.pi_logit.bias, 0.0)

    # overwrite
    def set_seed(self, seed):
        torch.manual_seed(seed)

    # overwrite
    def _critic_forward(self, inputs):
        x = inputs['obs']
        B = x.shape[0]
        x = x.view(B, -1)

        vf = self.act(self.vf_h1(x))
        vf = self.act(self.vf_h2(vf))
        vf = self.act(self.vf_h3(vf))
        vf = self.vf(vf)
        return vf

    # overwrite
    def _actor_forward(self, inputs):
        x = inputs['obs']
        B = x.shape[0]
        x = x.view(B, -1)

        pi = self.act(self.pi_h1(x))
        pi = self.act(self.pi_h2(pi))
        pi = self.act(self.pi_h3(pi))
        pi_logit = self.pi_logit(pi)

        if self.use_mask:
            mask = inputs['mask']
            assert(mask is not None)
            pi_logit -= (1-mask) * 1e30
        return pi_logit

    # overwrite
    def step(self, inputs):
        vf = self._critic_forward(inputs)
        pi_logit = self._actor_forward(inputs)
        handle = self.pd(pi_logit)
        action_select_func = {
            'rand': handle.noise_mode,
            'fixed': handle.mode,
            'sample': handle.sample,
        }
        if self.viz:
            action, logits_feature = action_select_func[self.action_type](viz=True)
        else:
            action = action_select_func[self.action_type](viz=False)
        neglogp = handle.neglogp(action, reduction='mean')
        ret = {
            'action': action,
            'value': vf,
            'neglogp': neglogp,
            'state': self.initial_state,
            'pi_logit': pi_logit,
        }
        if self.viz:
            ret['viz_feature'] = logits_feature
        return ret

    # overwrite
    def evaluate(self, inputs):
        vf = self._critic_forward(inputs)
        pi_logit = self._actor_forward(inputs)
        handle = self.pd(pi_logit)
        neglogp = handle.neglogp(inputs['action'], reduction='none')
        entropy = handle.entropy(reduction='mean')
        return {
            'value': vf,
            'neglogp': neglogp,
            'entropy': entropy,
            'state': self.initial_state,
            'pi_logit': pi_logit
        }

    # overwrite
    def value(self, inputs):
        return {'value': self._critic_forward(inputs)}


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


class PPOLSTM(ActorCriticBase):
    def __init__(self, ob_space, ac_space, unroll_length,
                 fc_dim=512, lstm_dim=512, seed=0):
        super(PPOLSTM, self).__init__()
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    # overwrite
    def set_seed(self, seed):
        torch.manual_seed(seed)

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
    model = PPOMLP(ob_space, ac_space)
    print(model)
    output_v = model(inputs, mode='value')
    print(output_v.shape)
    output_step = model(inputs, mode='step')
    for item in output_step:
        if item is not None:
            print(item.shape)
        else:
            print('none')


def test_mlp_policy_speed():
    from sc2learner.utils.time_helper import TimeWrapperCuda

    class T():
        pass

    def to_device(item, device):
        if isinstance(item, torch.nn.Module):
            return item.cuda()
        elif isinstance(item, torch.Tensor):
            return item.cuda()
        elif isinstance(item, dict):
            item = {k: to_device(item[k], device) for k in item.keys()}
            return item
    inputs = {}
    inputs['obs'] = torch.randn(2, 857)
    inputs['mask'] = torch.randn(2, 62)
    inputs['action'] = torch.ones(2).long()
    inputs = to_device(inputs, 'cuda')
    ob_space = torch.empty(857)
    ac_space = T()
    setattr(ac_space, 'n', 62)
    model = PPOMLP(ob_space, ac_space)
    model = to_device(model, 'cuda')
    model.use_mask = True
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    T = 100
    for _ in range(10):
        model(inputs, mode='value')
    time_dict = {'forward': [],
                 'backward': [],
                 'update': []}
    adv = torch.randn(2, 1).cuda()
    for t in range(T):
        TimeWrapperCuda.start_time()
        outputs = model(inputs, mode='evaluate')
        v, n, entropy = outputs['value'], outputs['neglogp'], outputs['entropy']
        new_v = v + torch.randn_like(v)
        new_n = n + torch.randn_like(n)
        new_v = v + torch.clamp(new_v - v, -0.1, 0.1)
        v_loss = (new_v - v) ** 2
        if t > 95:
            print(new_n.shape)
        ratio = torch.exp(n - new_n)
        pg_loss = -adv * ratio
        loss = pg_loss.mean() + v_loss.mean() - 0.01 * entropy
        approximate_kl = 0.5 * torch.pow(new_n - n, 2).mean()
        clipfrac = torch.abs(ratio - 1.0).gt(0.1).float().mean()
        time_dict['forward'].append(TimeWrapperCuda.end_time())

        TimeWrapperCuda.start_time()
        optimizer.zero_grad()
        loss.backward()
        time_dict['backward'].append(TimeWrapperCuda.end_time())

        TimeWrapperCuda.start_time()
        optimizer.step()
        time_dict['update'].append(TimeWrapperCuda.end_time())
        if t % 10 == 0:
            print(t)

    for k, v in time_dict.items():
        print(k, sum(v) / len(v))


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
    model = PPOLSTM(ob_space, ac_space, unroll_length=5)
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
    # test_mlp_policy()
    # test_lstm_policy()
    test_mlp_policy_speed()
