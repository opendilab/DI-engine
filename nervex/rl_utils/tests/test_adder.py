import pytest
import copy
from collections import deque
import numpy as np
import torch
from nervex.rl_utils import Adder


@pytest.mark.unittest
class TestAdder:

    def get_transition(self):
        return {
            'value': torch.randn(1),
            'reward': torch.rand(1),
            'other': np.random.randint(0, 10, size=(4, )),
            'obs': torch.randn(3),
            'done': False
        }

    def test_get_gae(self):
        adder = Adder(use_cuda=False, unroll_len=1)

        transitions = [self.get_transition() for _ in range(10)]
        last_value = torch.randn(1)
        output = adder.get_gae(transitions, last_value, gamma=0.99, gae_lambda=0.97)
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
        output1 = adder.get_gae_with_default_last_value(copy.deepcopy(transitions), True, gamma=0.99, gae_lambda=0.97)
        for i in range(len(output)):
            assert output[i]['adv'].ne(output1[i]['adv'])

        data = copy.deepcopy(transitions)
        data.append({'value': last_value})
        output2 = adder.get_gae_with_default_last_value(data, False, gamma=0.99, gae_lambda=0.97)
        for i in range(len(output)):
            assert output[i]['adv'].eq(output2[i]['adv'])

    def test_get_traj(self):
        adder = Adder(use_cuda=False, unroll_len=1)
        queue = deque(maxlen=8)
        for _ in range(8):
            queue.append(self.get_transition())
        traj = adder.get_traj(queue, traj_len=7, return_num=1)
        assert len(traj) == 7
        assert len(queue) == 2
        left_data = queue.popleft()
        for k in left_data:
            if k == 'done':
                continue
            assert (left_data[k] == traj[-1][k]).all()

        left_data['other'] = left_data['other'] + 1
        assert (left_data['other'] != traj[-1]['other']).all()

    def test_get_nstep_return_data(self):
        nstep = 3
        adder = Adder(use_cuda=False, unroll_len=1)
        data = [self.get_transition() for _ in range(10)]
        output_data = adder.get_nstep_return_data(data, nstep=nstep, traj_len=12)
        assert len(output_data) == 10
        for i, o in enumerate(output_data):
            assert o['reward'].shape == (nstep, )
            if i >= 10 - nstep + 1:
                assert o['done'] is True
                assert o['reward'][-(i - 10 + nstep):].sum() == 0

        data = [self.get_transition() for _ in range(12)]
        output_data = adder.get_nstep_return_data(data, nstep=nstep, traj_len=12)
        assert len(output_data) == 12 - nstep

    def test_get_train_sample(self):
        adder = Adder(use_cuda=False, unroll_len=1, last_fn_type='drop')
        data = [self.get_transition() for _ in range(10)]
        output = adder.get_train_sample(data)
        assert len(output) == 10

        adder = Adder(use_cuda=False, unroll_len=4, last_fn_type='drop')
        output = adder.get_train_sample(data)
        assert len(output) == 2
        for o in output:
            for v in o.values():
                assert len(v) == 4

        adder = Adder(use_cuda=False, unroll_len=4, last_fn_type='null_padding')
        output = adder.get_train_sample(data)
        assert len(output) == 3
        for o in output:
            for v in o.values():
                assert len(v) == 4
        assert output[-1]['done'] == [False, False, True, True]
        for i in range(1, 10 % 4 + 1):
            assert id(output[-1]['obs'][-i]) != id(output[-1]['obs'][0])

        adder = Adder(use_cuda=False, unroll_len=4, last_fn_type='last')
        output = adder.get_train_sample(data)
        assert len(output) == 3
        for o in output:
            for v in o.values():
                assert len(v) == 4
        miss_num = 4 - 10 % 4
        for i in range(10 % 4):
            assert id(output[-1]['obs'][i]) != id(output[-2]['obs'][miss_num + i])

        adder = Adder(use_cuda=False, unroll_len=11, last_fn_type='last')
        output = adder.get_train_sample(data)
        assert len(output) == 1
        assert len(output[0]['obs']) == 11
        assert output[-1]['done'][-1] is True
        assert output[-1]['done'][0] is False
        assert id(output[-1]['obs'][-1]) != id(output[-1]['obs'][0])
