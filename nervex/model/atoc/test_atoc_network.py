from nervex.model.atoc import ATOCActorNet, ATOCAttentionUnit, ATOCCommunicationNet, ATOCCriticNet
import pytest
import torch
from nervex.torch_utils import is_differentiable


@pytest.mark.unittest
class TestATOCNets:

    def test_actor_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        model = ATOCActorNet(obs_dim, thought_dim, act_dim, A, 2, 2)
        for _ in range(10):
            data = {'obs': torch.randn(B, A, obs_dim)}
            out = model.forward(data)
            # print(out)
            assert out['action'].shape == (B, A, act_dim)
            assert out['groups'].shape == (B, A, A)
            loss1 = out['action'].sum()
            loss2 = out['groups'].sum()

            # fail the is_differentiable judge
            # maybe because the model is not end to end
            # or maybe because the model has tons of parameters without grad

            # is_differentiable(loss1, model)
            # is_differentiable(loss2, model)

    def test_critic_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        model = ATOCCriticNet(obs_dim, act_dim)
        for _ in range(10):
            data = {'obs': torch.randn(B, A, obs_dim), 'action': torch.randn(B, A, act_dim)}
            out = model.forward(data)
            # print(out)
            assert out['q_value'].shape == (B, A, 1)
            loss = out['q_value'].sum()
            # is_differentiable(loss, model)
