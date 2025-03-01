import pytest
import numpy as np
import torch
# Import GRPO related functions
from ding.rl_utils.grpo import grpo_policy_data, grpo_policy_error


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
def test_grpo_policy_loss_with_mask(batch_size: int = 4, seq_length: int = 8, vocab_size: int = 1000):
    """Test GRPO policy loss calculation with mask"""
    # 1. Create test data
    logit_new = (torch.randn(batch_size, seq_length, vocab_size).requires_grad_(True))
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    logit_ref = logit_new + torch.randn_like(logit_new) * 0.2
    action = torch.randint(0, vocab_size, (batch_size, seq_length))
    adv = torch.randn(batch_size)
    weight = torch.ones(batch_size, seq_length)
    weight[:, -2:] = 0

    # 2. Create grpo_policy_data instance
    data = grpo_policy_data(
        logit_new=logit_new,  # Current policy output
        logit_old=logit_old,  # Old policy output
        logit_ref=logit_ref,  # Reference policy output
        action=action,  # Sampled tokens
        adv=adv,  # Advantage values
        weight=weight  # Attention mask
    )

    # 3. Calculate GRPO loss
    loss, info = grpo_policy_error(
        data=data,
        clip_ratio=0.2,  # PPO clipping ratio
        beta=0.1  # KL divergence weight
    )

    # 4. Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Ensure scalar output
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # 5. Test gradients
    assert logit_new.grad is None
    loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    assert 'approx_kl' in info._asdict()
    assert 'clipfrac' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])


@pytest.mark.unittest
def test_grpo_policy_loss_without_mask(batch_size: int = 4, seq_length: int = 8, vocab_size: int = 1000):
    """Test GRPO policy loss calculation without mask"""
    # 1. Create test data
    logit_new = torch.randn(batch_size, seq_length, vocab_size).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    logit_ref = logit_new + torch.randn_like(logit_new) * 0.2
    action = torch.randint(0, vocab_size, (batch_size, seq_length))
    adv = torch.randn(batch_size)

    # 2. Create grpo_policy_data instance
    data = grpo_policy_data(
        logit_new=logit_new,  # Current policy output
        logit_old=logit_old,  # Old policy output
        logit_ref=logit_ref,  # Reference policy output
        action=action,  # Sampled tokens
        adv=adv,  # Advantage values
        weight=None  # No mask
    )

    # 3. Calculate GRPO loss
    loss, info = grpo_policy_error(
        data=data,
        clip_ratio=0.2,  # PPO clipping ratio
        beta=0.1  # KL divergence weight
    )

    # 4. Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Ensure scalar output
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # 5. Test gradients
    assert logit_new.grad is None
    loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    # 6. Verify metrics
    assert 'approx_kl' in info._asdict()
    assert 'clipfrac' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])
