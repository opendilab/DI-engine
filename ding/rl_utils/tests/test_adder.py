import pytest
import copy
from collections import deque
import numpy as np
import torch
from ding.rl_utils import get_gae, get_gae_with_default_last_value, get_nstep_return_data, get_train_sample


@pytest.mark.unittest
class TestAdder:

    def get_transition(self):
        return {
            'value': torch.randn(1),
            'reward': torch.rand(1),
            'action': torch.rand(3),
            'other': np.random.randint(0, 10, size=(4, )),
            'obs': torch.randn(3),
            'done': False
        }

    def get_transition_multi_agent(self):
        return {
            'value': torch.randn(1, 8),
            'reward': torch.rand(1, 1),
            'action': torch.rand(3),
            'other': np.random.randint(0, 10, size=(4, )),
            'obs': torch.randn(3),
            'done': False
        }

    def test_get_gae(self):
        transitions = deque([self.get_transition() for _ in range(10)])
        last_value = torch.randn(1)
        output = get_gae(transitions, last_value, gamma=0.99, gae_lambda=0.97, cuda=False)
        for i in range(len(output)):
            o = output[i]
            assert 'adv' in o.keys()
            for k, v in o.items():
                if k == 'adv':
                    assert isinstance(v, torch.Tensor)
                    assert v.shape == (1, )
                else:
                    if k == 'done':
                        assert v == transitions[i][k]
                    else:
                        assert (v == transitions[i][k]).all()
        output1 = get_gae_with_default_last_value(
            copy.deepcopy(transitions), True, gamma=0.99, gae_lambda=0.97, cuda=False
        )
        for i in range(len(output)):
            assert output[i]['adv'].ne(output1[i]['adv'])

        data = copy.deepcopy(transitions)
        data.append({'value': last_value})
        output2 = get_gae_with_default_last_value(data, False, gamma=0.99, gae_lambda=0.97, cuda=False)
        for i in range(len(output)):
            assert output[i]['adv'].eq(output2[i]['adv'])

    def test_get_gae_multi_agent(self):
        transitions = deque([self.get_transition_multi_agent() for _ in range(10)])
        last_value = torch.randn(1, 8)
        output = get_gae(transitions, last_value, gamma=0.99, gae_lambda=0.97, cuda=False)
        for i in range(len(output)):
            o = output[i]
            assert 'adv' in o.keys()
            for k, v in o.items():
                if k == 'adv':
                    assert isinstance(v, torch.Tensor)
                    assert v.shape == (
                        1,
                        8,
                    )
                else:
                    if k == 'done':
                        assert v == transitions[i][k]
                    else:
                        assert (v == transitions[i][k]).all()
        output1 = get_gae_with_default_last_value(
            copy.deepcopy(transitions), True, gamma=0.99, gae_lambda=0.97, cuda=False
        )
        for i in range(len(output)):
            for j in range(output[i]['adv'].shape[1]):
                assert output[i]['adv'][0][j].ne(output1[i]['adv'][0][j])

        data = copy.deepcopy(transitions)
        data.append({'value': last_value})
        output2 = get_gae_with_default_last_value(data, False, gamma=0.99, gae_lambda=0.97, cuda=False)
        for i in range(len(output)):
            for j in range(output[i]['adv'].shape[1]):
                assert output[i]['adv'][0][j].eq(output2[i]['adv'][0][j])

    def test_get_nstep_return_data(self):
        nstep = 3
        data = deque([self.get_transition() for _ in range(10)])
        output_data = get_nstep_return_data(data, nstep=nstep)
        assert len(output_data) == 10
        for i, o in enumerate(output_data):
            assert o['reward'].shape == (nstep, )
            if i >= 10 - nstep + 1:
                assert o['done'] is data[-1]['done']
                assert o['reward'][-(i - 10 + nstep):].sum() == 0

        data = deque([self.get_transition() for _ in range(12)])
        output_data = get_nstep_return_data(data, nstep=nstep)
        assert len(output_data) == 12

    def test_get_train_sample(self):
        data = [self.get_transition() for _ in range(10)]
        output = get_train_sample(data, unroll_len=1, last_fn_type='drop')
        assert len(output) == 10

        output = get_train_sample(data, unroll_len=4, last_fn_type='drop')
        assert len(output) == 2
        for o in output:
            for v in o.values():
                assert len(v) == 4

        output = get_train_sample(data, unroll_len=4, last_fn_type='null_padding')
        assert len(output) == 3
        for o in output:
            for v in o.values():
                assert len(v) == 4
        assert output[-1]['done'] == [False, False, True, True]
        for i in range(1, 10 % 4 + 1):
            assert id(output[-1]['obs'][-i]) != id(output[-1]['obs'][0])

        output = get_train_sample(data, unroll_len=4, last_fn_type='last')
        assert len(output) == 3
        for o in output:
            for v in o.values():
                assert len(v) == 4
        miss_num = 4 - 10 % 4
        for i in range(10 % 4):
            assert id(output[-1]['obs'][i]) != id(output[-2]['obs'][miss_num + i])

        output = get_train_sample(data, unroll_len=11, last_fn_type='last')
        assert len(output) == 1
        assert len(output[0]['obs']) == 11
        assert output[-1]['done'][-1] is True
        assert output[-1]['done'][0] is False
        assert id(output[-1]['obs'][-1]) != id(output[-1]['obs'][0])


test = TestAdder()
test.test_get_gae_multi_agent()
