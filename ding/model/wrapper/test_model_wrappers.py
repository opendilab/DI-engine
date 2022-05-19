import copy
from copy import deepcopy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from ditk import logging

from ding.torch_utils import get_lstm
from ding.torch_utils.network.gtrxl import GTrXL
from ding.model import model_wrap, register_wrapper, IModelWrapper, BaseModelWrapper


class TempMLP(torch.nn.Module):

    def __init__(self):
        super(TempMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x


class ActorMLP(torch.nn.Module):

    def __init__(self):
        super(ActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Softmax()

    def forward(self, inputs, tmp=0):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.out(x)
        ret = {'logit': x, 'tmp': tmp, 'action': x + torch.rand_like(x)}
        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class HybridActorMLP(torch.nn.Module):

    def __init__(self):
        super(HybridActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Softmax()

        self.fc2_cont = nn.Linear(4, 6)
        self.act_cont = nn.ReLU()

    def forward(self, inputs, tmp=0):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x_ = self.act(x)

        x = self.fc2(x_)
        x = self.act(x)
        x_disc = self.out(x)

        x = self.fc2_cont(x_)
        x_cont = self.act_cont(x)

        ret = {'logit': x_disc, 'action_args': x_cont, 'tmp': tmp}

        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class HybridReparamActorMLP(torch.nn.Module):

    def __init__(self):
        super(HybridReparamActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Softmax()

        self.fc2_cont_mu = nn.Linear(4, 6)
        self.act_cont_mu = nn.ReLU()

        self.fc2_cont_sigma = nn.Linear(4, 6)
        self.act_cont_sigma = nn.ReLU()

    def forward(self, inputs, tmp=0):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x_ = self.act(x)

        x = self.fc2(x_)
        x = self.act(x)
        x_disc = self.out(x)

        x = self.fc2_cont_mu(x_)
        x_cont_mu = self.act_cont_mu(x)

        x = self.fc2_cont_sigma(x_)
        x_cont_sigma = self.act_cont_sigma(x) + 1e-8

        ret = {'logit': {'action_type': x_disc, 'action_args': {'mu': x_cont_mu, 'sigma': x_cont_sigma}}, 'tmp': tmp}

        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class ReparamActorMLP(torch.nn.Module):

    def __init__(self):
        super(ReparamActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Softmax()

        self.fc2_cont_mu = nn.Linear(4, 6)
        self.fc2_cont_sigma = nn.Linear(4, 6)

    def forward(self, inputs, tmp=0):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x_ = self.act(x)

        x = self.fc2_cont_mu(x_)
        x_cont_mu = self.act(x)

        x = self.fc2_cont_sigma(x_)
        x_cont_sigma = self.act(x) + 1e-8

        ret = {'logit': {'mu': x_cont_mu, 'sigma': x_cont_sigma}, 'tmp': tmp}

        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class DeterministicActorMLP(torch.nn.Module):

    def __init__(self):
        super(DeterministicActorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.act = nn.ReLU()

        self.fc2_cont_mu = nn.Linear(4, 6)
        self.act_cont_mu = nn.ReLU()

    def forward(self, inputs):
        x = self.fc1(inputs['obs'])
        x = self.bn1(x)
        x_ = self.act(x)

        x = self.fc2_cont_mu(x_)
        x_cont_mu = self.act_cont_mu(x)

        ret = {
            'logit': {
                'mu': x_cont_mu,
            }
        }

        if 'mask' in inputs:
            ret['action_mask'] = inputs['mask']
        return ret


class TempLSTM(torch.nn.Module):

    def __init__(self):
        super(TempLSTM, self).__init__()
        self.model = get_lstm(lstm_type='pytorch', input_size=36, hidden_size=32, num_layers=2, norm_type=None)

    def forward(self, data):
        output, next_state = self.model(data['f'], data['prev_state'], list_next_state=True)
        return {'output': output, 'next_state': next_state}


@pytest.fixture(scope='function')
def setup_model():
    return torch.nn.Linear(3, 6)


@pytest.mark.unittest
class TestModelWrappers:

    def test_hidden_state_wrapper(self):

        model = TempLSTM()
        state_num = 4
        model = model_wrap(model, wrapper_name='hidden_state', state_num=state_num, save_prev_state=True)
        model.reset()
        data = {'f': torch.randn(2, 4, 36)}
        output = model.forward(data)
        assert output['output'].shape == (2, state_num, 32)
        assert len(output['prev_state']) == 4
        assert output['prev_state'][0]['h'].shape == (2, 1, 32)
        for item in model._state.values():
            assert isinstance(item, dict) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item.values())

        data = {'f': torch.randn(2, 3, 36)}
        data_id = [0, 1, 3]
        output = model.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 3, 32)
        assert all([len(s) == 2 for s in output['prev_state']])
        for item in model._state.values():
            assert isinstance(item, dict) and len(item) == 2
            assert all(t.shape == (2, 1, 32) for t in item.values())

        data = {'f': torch.randn(2, 2, 36)}
        data_id = [0, 1]
        output = model.forward(data, data_id=data_id)
        assert output['output'].shape == (2, 2, 32)

        assert all([isinstance(s, dict) and len(s) == 2 for s in model._state.values()])
        model.reset()
        assert all([isinstance(s, type(None)) for s in model._state.values()])

    def test_target_network_wrapper(self):

        model = TempMLP()
        target_model = deepcopy(model)
        target_model2 = deepcopy(model)
        target_model = model_wrap(target_model, wrapper_name='target', update_type='assign', update_kwargs={'freq': 2})
        model = model_wrap(model, wrapper_name='base')
        register_wrapper('abstract', IModelWrapper)
        assert all([hasattr(target_model, n) for n in ['reset', 'forward', 'update']])
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict(), direct=True)
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12
        model.reset()
        target_model.reset()

        inputs = torch.randn(2, 3)
        model.train()
        target_model.train()
        output = model.forward(inputs)
        with torch.no_grad():
            output_target = target_model.forward(inputs)
        assert output.eq(output_target).sum() == 2 * 6
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict())
        assert model.fc1.weight.ne(target_model.fc1.weight).sum() == 12
        target_model.update(model.state_dict())
        assert model.fc1.weight.eq(target_model.fc1.weight).sum() == 12
        # test real reset update_count
        assert target_model._update_count != 0
        target_model.reset()
        assert target_model._update_count != 0
        target_model.reset(target_update_count=0)
        assert target_model._update_count == 0

        target_model2 = model_wrap(
            target_model2, wrapper_name='target', update_type='momentum', update_kwargs={'theta': 0.01}
        )
        target_model2.update(model.state_dict(), direct=True)
        assert model.fc1.weight.eq(target_model2.fc1.weight).sum() == 12
        model.fc1.weight.data = torch.randn_like(model.fc1.weight)
        old_state_dict = target_model2.state_dict()
        target_model2.update(model.state_dict())
        assert target_model2.fc1.weight.data.eq(
            old_state_dict['fc1.weight'] * (1 - 0.01) + model.fc1.weight.data * 0.01
        ).all()

    def test_eps_greedy_wrapper(self):
        model = ActorMLP()
        model = model_wrap(model, wrapper_name='eps_greedy_sample')
        model.eval()
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold)
        assert output['tmp'] == 0
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1)
            assert isinstance(output, dict)
        assert output['tmp'] == 1

    def test_multinomial_sample_wrapper(self):
        model = model_wrap(ActorMLP(), wrapper_name='multinomial_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        assert output['action'].shape == (4, )
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = model.forward(data)
        assert output['action'].shape == (4, )

    def test_eps_greedy_multinomial_wrapper(self):
        model = ActorMLP()
        model = model_wrap(model, wrapper_name='eps_greedy_multinomial_sample')
        model.eval()
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold, alpha=0.2)
        assert output['tmp'] == 0
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1, alpha=0.2)
            assert isinstance(output, dict)
        assert output['tmp'] == 1

    def test_hybrid_eps_greedy_wrapper(self):
        model = HybridActorMLP()
        model = model_wrap(model, wrapper_name='hybrid_eps_greedy_sample')
        model.eval()
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold)
        # logit = output['logit']
        # assert output['action']['action_type'].eq(logit.argmax(dim=-1)).all()
        assert isinstance(output['action']['action_args'],
                          torch.Tensor) and output['action']['action_args'].shape == (4, 6)
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1)
            assert isinstance(output, dict)

    def test_hybrid_eps_greedy_multinomial_wrapper(self):
        model = HybridActorMLP()
        model = model_wrap(model, wrapper_name='hybrid_eps_greedy_multinomial_sample')
        model.eval()
        eps_threshold = 0.5
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold)
        assert isinstance(output['logit'], torch.Tensor) and output['logit'].shape == (4, 6)
        assert isinstance(output['action']['action_type'],
                          torch.Tensor) and output['action']['action_type'].shape == (4, )
        assert isinstance(output['action']['action_args'],
                          torch.Tensor) and output['action']['action_args'].shape == (4, 6)
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1)
            assert isinstance(output, dict)

    def test_hybrid_reparam_multinomial_wrapper(self):
        model = HybridReparamActorMLP()
        model = model_wrap(model, wrapper_name='hybrid_reparam_multinomial_sample')
        model.eval()
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data)
        assert isinstance(output['logit'], dict) and output['logit']['action_type'].shape == (4, 6)
        assert isinstance(output['logit']['action_args'], dict) and output['logit']['action_args']['mu'].shape == (
            4, 6
        ) and output['logit']['action_args']['sigma'].shape == (4, 6)
        assert isinstance(output['action']['action_type'],
                          torch.Tensor) and output['action']['action_type'].shape == (4, )
        assert isinstance(output['action']['action_args'],
                          torch.Tensor) and output['action']['action_args'].shape == (4, 6)
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, tmp=1)
            assert isinstance(output, dict)

    def test_argmax_sample_wrapper(self):
        model = model_wrap(ActorMLP(), wrapper_name='argmax_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        logit = output['logit']
        assert output['action'].eq(logit.argmax(dim=-1)).all()
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = model.forward(data)
        logit = output['logit'].sub(1e8 * (1 - data['mask']))
        assert output['action'].eq(logit.argmax(dim=-1)).all()

    def test_hybrid_argmax_sample_wrapper(self):
        model = model_wrap(HybridActorMLP(), wrapper_name='hybrid_argmax_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        logit = output['logit']
        assert output['action']['action_type'].eq(logit.argmax(dim=-1)).all()
        assert isinstance(output['action']['action_args'],
                          torch.Tensor) and output['action']['action_args'].shape == (4, 6)
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        output = model.forward(data)
        logit = output['logit'].sub(1e8 * (1 - data['mask']))
        assert output['action']['action_type'].eq(logit.argmax(dim=-1)).all()
        assert output['action']['action_args'].shape == (4, 6)

    def test_hybrid_deterministic_argmax_sample_wrapper(self):
        model = model_wrap(HybridReparamActorMLP(), wrapper_name='hybrid_deterministic_argmax_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        assert output['action']['action_type'].eq(output['logit']['action_type'].argmax(dim=-1)).all()
        assert isinstance(output['action']['action_args'],
                          torch.Tensor) and output['action']['action_args'].shape == (4, 6)
        assert output['action']['action_args'].eq(output['logit']['action_args']['mu']).all

    def test_deterministic_sample_wrapper(self):
        model = model_wrap(DeterministicActorMLP(), wrapper_name='deterministic_sample')
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        assert output['action'].eq(output['logit']['mu']).all()
        assert isinstance(output['action'], torch.Tensor) and output['action'].shape == (4, 6)

    def test_reparam_wrapper(self):
        model = ReparamActorMLP()
        model = model_wrap(model, wrapper_name='reparam_sample')
        model.eval()
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data)
        assert isinstance(output['logit'],
                          dict) and output['logit']['mu'].shape == (4, 6) and output['logit']['sigma'].shape == (4, 6)
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, tmp=1)
            assert isinstance(output, dict)

    def test_eps_greedy_wrapper_with_list_eps(self):
        model = ActorMLP()
        model = model_wrap(model, wrapper_name='eps_greedy_sample')
        model.eval()
        eps_threshold = {i: 0.5 for i in range(4)}  # for NGU
        data = {'obs': torch.randn(4, 3), 'mask': torch.randint(0, 2, size=(4, 6))}
        with torch.no_grad():
            output = model.forward(data, eps=eps_threshold)
        assert output['tmp'] == 0
        for i in range(10):
            if i == 5:
                data.pop('mask')
            with torch.no_grad():
                output = model.forward(data, eps=eps_threshold, tmp=1)
            assert isinstance(output, dict)
        assert output['tmp'] == 1

    def test_action_noise_wrapper(self):
        model = model_wrap(
            ActorMLP(),
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_range={
                'min': -0.1,
                'max': 0.1
            },
            action_range={
                'min': -0.05,
                'max': 0.05
            }
        )
        data = {'obs': torch.randn(4, 3)}
        output = model.forward(data)
        action = output['action']
        assert action.shape == (4, 6)
        assert action.eq(action.clamp(-0.05, 0.05)).all()

    def test_transformer_input_wrapper(self):
        seq_len, bs, obs_shape = 8, 8, 32
        emb_dim = 64
        model = GTrXL(input_dim=obs_shape, embedding_dim=emb_dim)
        model = model_wrap(model, wrapper_name='transformer_input', seq_len=seq_len)
        obs = []
        for i in range(seq_len + 1):
            obs.append(torch.randn((bs, obs_shape)))
        out = model.forward(obs[0], only_last_logit=False)
        assert out['logit'].shape == (seq_len, bs, emb_dim)
        assert out['input_seq'].shape == (seq_len, bs, obs_shape)
        assert sum(out['input_seq'][1:].flatten()) == 0
        for i in range(1, seq_len - 1):
            out = model.forward(obs[i])
        assert out['logit'].shape == (bs, emb_dim)
        assert out['input_seq'].shape == (seq_len, bs, obs_shape)
        assert sum(out['input_seq'][seq_len - 1:].flatten()) == 0
        assert sum(out['input_seq'][:seq_len - 1].flatten()) != 0
        out = model.forward(obs[seq_len - 1])
        prev_memory = torch.clone(out['input_seq'])
        out = model.forward(obs[seq_len])
        assert torch.all(torch.eq(out['input_seq'][seq_len - 2], prev_memory[seq_len - 1]))
        # test update of single batches in the memory
        model.reset(data_id=[0, 5])  # reset memory batch in position 0 and 5
        assert sum(model.obs_memory[:, 0].flatten()) == 0 and sum(model.obs_memory[:, 5].flatten()) == 0
        assert sum(model.obs_memory[:, 1].flatten()) != 0
        assert model.memory_idx[0] == 0 and model.memory_idx[5] == 0 and model.memory_idx[1] == seq_len
        # test reset
        model.reset()
        assert model.obs_memory is None

    def test_transformer_memory_wrapper(self):
        seq_len, bs, obs_shape = 12, 8, 32
        layer_num, memory_len, emb_dim = 3, 4, 4
        model = GTrXL(input_dim=obs_shape, embedding_dim=emb_dim, memory_len=memory_len, layer_num=layer_num)
        model1 = model_wrap(model, wrapper_name='transformer_memory', batch_size=bs)
        model2 = model_wrap(model, wrapper_name='transformer_memory', batch_size=bs)
        inputs1 = torch.randn((seq_len, bs, obs_shape))
        out = model1.forward(inputs1)
        new_memory1 = model1.memory
        inputs2 = torch.randn((seq_len, bs, obs_shape))
        out = model2.forward(inputs2)
        new_memory2 = model2.memory
        assert not torch.all(torch.eq(new_memory1, new_memory2))
        model1.reset(data_id=[0, 5])
        assert sum(model1.memory[:, :, 0].flatten()) == 0 and sum(model1.memory[:, :, 5].flatten()) == 0
        assert sum(model1.memory[:, :, 1].flatten()) != 0
        model1.reset()
        assert sum(model1.memory.flatten()) == 0

        seq_len, bs, obs_shape = 8, 8, 32
        layer_num, memory_len, emb_dim = 3, 20, 4
        model = GTrXL(input_dim=obs_shape, embedding_dim=emb_dim, memory_len=memory_len, layer_num=layer_num)
        model = model_wrap(model, wrapper_name='transformer_memory', batch_size=bs)
        inputs1 = torch.randn((seq_len, bs, obs_shape))
        out = model.forward(inputs1)
        new_memory1 = model.memory
        inputs2 = torch.randn((seq_len, bs, obs_shape))
        out = model.forward(inputs2)
        new_memory2 = model.memory
        print(new_memory1.shape, inputs1.shape)
        assert sum(new_memory1[:, -8:].flatten()) != 0
        assert sum(new_memory1[:, :-8].flatten()) == 0
        assert sum(new_memory2[:, -16:].flatten()) != 0
        assert sum(new_memory2[:, :-16].flatten()) == 0
        assert torch.all(torch.eq(new_memory1[:, -8:], new_memory2[:, -16:-8]))
