import pytest
import numpy as np
import torch
from ding.rl_utils import ppo_policy_data, ppo_value_data, ppo_policy_error, ppo_value_error


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_length():
    return 8


@pytest.fixture
def dictionary_num():
    return 1000


@pytest.mark.unittest
def test_policy_loss_without_mask(batch_size: int, seq_length: int, dictionary_num: int):
    # Create test data
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    logit_pretrained = logit_new + torch.randn_like(logit_new) * 0.1
    action = torch.randint(0, 10, (batch_size, seq_length))
    advantages = torch.randn(batch_size, seq_length)

    # Compute loss
    data = ppo_policy_data(logit_new, logit_old, action, advantages, weight=None, logit_pretrained=logit_pretrained)
    loss, info = ppo_policy_error(data, clip_ratio=0.2, entropy_bonus=False)

    # Verify output
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # scalar
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert all([np.isscalar(i) for i in info])


@pytest.mark.unittest
def test_policy_loss_with_mask(batch_size: int, seq_length: int, dictionary_num: int):
    # Create test data
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    logit_pretrained = logit_new + torch.randn_like(logit_new) * 0.1
    action = torch.randint(0, 10, (batch_size, seq_length))
    advantages = torch.randn(batch_size, seq_length)
    action_mask = torch.ones(batch_size, seq_length)
    action_mask[:, -2:] = 0  # Set last two timesteps as padding

    # Compute loss
    data = ppo_policy_data(
        logit_new, logit_old, action, advantages, weight=action_mask, logit_pretrained=logit_pretrained
    )
    loss, info = ppo_policy_error(data, clip_ratio=0.2, entropy_bonus=False)

    # Verify output
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # scalar
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert all([np.isscalar(i) for i in info])


@pytest.mark.unittest
def test_value_loss(batch_size: int, seq_length: int):
    # Create test data
    values = torch.randn(batch_size, seq_length).requires_grad_(True)
    old_values = values + torch.randn_like(values) * 0.1
    returns = torch.randn(batch_size, seq_length)

    # Compute loss
    data = ppo_value_data(values, old_values, returns, weight=None)
    value_loss = ppo_value_error(data, clip_ratio=0.2, use_value_clip=True)

    # Verify output
    assert isinstance(value_loss, torch.Tensor)
    assert value_loss.shape == torch.Size([])
    assert not torch.isnan(value_loss)
    assert not torch.isinf(value_loss)
    assert values.grad is None
    value_loss.backward()
    assert isinstance(values.grad, torch.Tensor)
