import pytest
import numpy as np
import torch
from nervex.rl_utils import Adder


@pytest.mark.unittest
class TestAdder:

    def get_transition(self):
        return {'value': torch.randn(1), 'reward': torch.rand(1), 'other': np.random.randint(0, 10, size=(4, ))}

    def test_get_gae(self):
        adder = Adder(use_cuda=False)

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
                    assert (v == transitions[i][k]).all()
