import pytest
import torch
from ding.model.template.atoc import ATOCActorNet, ATOC
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
class TestATOC:

    @pytest.mark.tmp
    def test_actor_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        torch.autograd.set_detect_anomaly(True)
        model = ATOCActorNet(obs_dim, thought_dim, act_dim, A, True, 2, initiator_threshold=0.001)
        for i in range(10):
            out = model.forward(torch.randn(B, A, obs_dim))
            assert out['action'].shape == (B, A, act_dim)
            assert out['group'].shape == (B, A, A)
            loss1 = out['action'].sum()
            if i == 0:
                is_differentiable(loss1, model, print_instead=True)
            else:
                loss1.backward()

    def test_qac_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        model = ATOC(obs_dim, act_dim, thought_dim, A, True, 2, 2)

        # test basic forward path

        optimize_critic = torch.optim.SGD(model.critic.parameters(), 0.1)
        obs = torch.randn(B, A, obs_dim)
        act = torch.rand(B, A, act_dim)
        out = model({'obs': obs, 'action': act}, mode='compute_critic')
        assert out['q_value'].shape == (B, A)
        q_loss = out['q_value'].sum()
        q_loss.backward()
        optimize_critic.step()

        out = model(obs, mode='compute_actor', get_delta_q=True)
        assert out['delta_q'].shape == (B, A)
        assert out['initiator_prob'].shape == (B, A)
        assert out['is_initiator'].shape == (B, A)
        optimizer_act = torch.optim.SGD(model.actor.parameters(), 0.1)
        optimizer_att = torch.optim.SGD(model.actor.attention.parameters(), 0.1)

        obs = torch.randn(B, A, obs_dim)
        delta_q = model(obs, mode='compute_actor', get_delta_q=True)
        attention_loss = model(delta_q, mode='optimize_actor_attention')
        optimizer_att.zero_grad()
        loss = attention_loss['loss']
        loss.backward()
        optimizer_att.step()

        weights = dict(model.actor.named_parameters())
        output = model(obs, mode='compute_actor')
        output['obs'] = obs
        q_loss = model(output, mode='compute_critic')
        loss = q_loss['q_value'].sum()
        before_update_weights = model.actor.named_parameters()
        optimizer_act.zero_grad()

        loss.backward()
        optimizer_act.step()
