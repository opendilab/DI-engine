import torch
import numpy as np
import pytest

from nervex.model.rnn_actor import RnnActorNetwork
from nervex.torch_utils import is_differentiable

B = 4
T = 6
embedding_dim = 32
# action_dim_args = [(6, ), [4, 8], [
#     1,
# ]]
action_dim_args = [1, 2, 4]
rnn_type_args = ['lstm', 'gru']


@pytest.mark.unittest
@pytest.mark.parametrize('action_dim', action_dim_args)
@pytest.mark.parametrize('rnn_type', rnn_type_args)
class TestRnnActorNet:

    def output_check(self, model, outputs, action_dim):
        if isinstance(action_dim, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_dim):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_rnn_actor_net(self, action_dim, rnn_type):
        N = 32
        data = torch.randn(T, B, N)
        model = RnnActorNetwork((N, ), action_dim, embedding_dim, rnn_type=rnn_type)
        prev_state = [None for _ in range(B)]
        for t in range(T):
            inputs = {'obs': data[t], 'prev_state': prev_state}
            outputs = model(inputs)
            logit, prev_state = outputs['logit'], outputs['next_state']
            assert len(prev_state) == B
            if rnn_type == 'lstm':
                assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
        # test the last step can backward correctly
        self.output_check(model, logit, action_dim)
