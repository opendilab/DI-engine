import pytest
import torch
import math
from sc2learner.agent.model.alphastar.head import ActionTypeHead, DelayHead, QueuedHead
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

    def test_delay_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.delay_head
        model = DelayHead(handle)
        # input_dim: 1024
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        # no delay
        output, delay, embedding = model(autoregressive_embedding)
        assert output.shape == (B, 1)
        assert output.dtype == torch.float
        assert delay.shape == (B, )
        assert delay.dtype == torch.long
        assert embedding.shape == (B, handle.input_dim)

        loss = output.mean() + embedding.mean()
        is_differentiable(loss, model)
        # indicated delay
        delay = torch.randint(0, int(model.delay_max_range), size=(B, ))
        output, delay_pred, embedding = model(autoregressive_embedding, delay)
        assert delay_pred.dtype == torch.long
        assert delay_pred.shape == (B, )
        assert delay_pred.eq(delay).sum() == B

    def test_queued_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.queued_head
        model = QueuedHead(handle)
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        # no queued
        output, queued, embedding = model(autoregressive_embedding, temperature=0.8)
        assert output.shape == (B, handle.queued_dim)
        assert output.dtype == torch.float
        assert queued.shape == (B, )
        assert queued.dtype == torch.long
        assert embedding.shape == (B, handle.input_dim)
        # indicated queued
        queued = torch.randint(0, handle.queued_dim, size=(B, ))
        output, queued_pred, embedding = model(autoregressive_embedding, queued=queued)
        assert queued_pred.eq(queued).sum() == B
