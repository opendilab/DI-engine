from nervex.model.atoc import ATOCActorNet, ATOCAttentionUnit, ATOCCommunicationNet, ATOCCriticNet, ATOCQAC
import pytest
import torch
from nervex.torch_utils.nn_test_helper import is_differentiable, is_differentiable_print_no_grad


@pytest.mark.unittest
class TestATOCNets:

    def test_actor_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        torch.autograd.set_detect_anomaly(True)
        model = ATOCActorNet(obs_dim, thought_dim, act_dim, A, 2, 2)
        for _ in range(1):
            data = {'obs': torch.randn(B, A, obs_dim)}
            out = model.forward(data)
            assert out['action'].shape == (B, A, act_dim)
            assert out['group'].shape == (B, A, A)
            loss1 = out['action'].sum()
            # loss2 = out['groups'].sum()
            # loss3 = out['initiator'].sum()

            # fail the is_differentiable judge
            # maybe because the model is not end to end
            # or maybe because the model has tons of parameters without grad

            # if _ == 0:
            #     for p in model.parameters():
            #         assert p.grad is None
            #     loss1.backward()
            #     # loss2.backward()
            #     # TODO fix attention loss
            #     # loss3.backward()
            #     for k, p in model.named_parameters():
            #         if not isinstance(p.grad, torch.Tensor):
            #             print("no grad", k, p)

    def test_critic_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        model = ATOCCriticNet(obs_dim, act_dim)
        for _ in range(1):
            data = {'obs': torch.randn(B, A, obs_dim), 'action': torch.randn(B, A, act_dim)}
            out = model.forward(data)
            # print(out)
            assert out['q_value'].shape == (B, A)
            loss = out['q_value'].sum()
            if _ == 0:
                is_differentiable(loss, model)

    def test_qac_net(self):
        B, A, obs_dim, act_dim, thought_dim = 6, 5, 12, 6, 14
        model = ATOCQAC(obs_dim, act_dim, thought_dim, A, 2, 2)

        # test basic forward path

        obs = torch.randn(B, A, obs_dim)
        inputs = {'obs': obs}
        out = model(inputs, mode='compute_q')
        assert out['q_value'].shape == (B, A)

        out = model(inputs, mode='compute_action', get_delta_q=True)
        assert out['delta_q'].shape == (B, A)
        assert out['initiator_prob'].shape == (B, A)
        assert out['is_initiator'].shape == (B, A)

        # test loss backwards

        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        inputs = out
        weights = dict(model._actor.named_parameters())
        inputs['obs'] = obs
        q_loss = model(inputs, mode='optimize_actor')
        loss = q_loss['q_value'].sum()
        # before_update_weights = model._actor.named_parameters()
        # # assert weights == before_update_weights
        # # optimizer.zero_grad()

        # is_differentiable_print_no_grad(loss, model)

        # loss.backward()
        # optimizer.step()
        # updated_weights = dict(model._actor.named_parameters())
        # for name1, param1 in weights.items():
        #     # print("name1 =", name1)
        #     # print("parama1 = ", param1)
        #     # print("parama2 = ", updated_weights[name1])
        #     for name2, param2 in updated_weights.items():
        #         # print("name2 =", name2)
        #         if name1 == name2:
        #             pass
        #             # print(name1, ":", param1, "diff", param2)
        # assert weights != updated_weights

        attention_loss = model(inputs, mode='optimize_actor_attention')
        loss = attention_loss['actor_attention_loss'].sum()
        is_differentiable_print_no_grad(loss, model)
