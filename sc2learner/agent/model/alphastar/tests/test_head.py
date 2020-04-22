import pytest
import torch
from sc2learner.agent.model.alphastar.head import ActionTypeHead
from sc2learner.agent.model.alphastar.tests.conftest import is_differentiable


@pytest.mark.unittest
class TestHead:
    def test_action_type_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.action_type_head
        model = ActionTypeHead(handle)

        lstm_output = torch.randn(B, handle.input_dim)
        scalar_context = torch.randn(B, handle.context_dim)
        mask = torch.ones(B, handle.action_num)
        # no action_type
        logits, action, embedding = model(lstm_output, scalar_context, mask)
        assert logits.shape == (B, handle.action_num)
        assert action.shape == (B, )
        assert action.dtype == torch.long
        assert embedding.shape == (B, handle.gate_dim)
        loss = logits.mean() + embedding.mean()
        is_differentiable(loss, model)

        # indicated action_type
        action_type = torch.randint(0, handle.action_num, size=(B, ))
        logits, action_type_output, embedding = model(lstm_output, scalar_context, mask, action_type=action_type)
        assert logits.shape == (B, handle.action_num)
        assert embedding.shape == (B, handle.gate_dim)
        assert action_type_output.dtype == torch.long
        assert action_type_output.eq(action_type).sum() == B
