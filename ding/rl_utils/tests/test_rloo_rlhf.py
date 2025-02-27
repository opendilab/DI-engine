import pytest
import numpy as np
import torch
from ding.rl_utils.rloo import (rloo_policy_data, rloo_policy_error)


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
def test_rloo_policy_loss_without_mask(batch_size, seq_length, dictionary_num):
    """Test RLOO policy loss calculation without mask"""
    # Create test data
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    action = torch.randint(0, dictionary_num, (batch_size, seq_length))
    reward = torch.randn(batch_size)

    # Calculate loss
    data = rloo_policy_data(logit_new=logit_new, logit_old=logit_old, action=action, reward=reward, weight=None)
    loss, info = rloo_policy_error(data, clip_ratio=0.2)

    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Ensure scalar output
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert logit_new.grad is None
    loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert all([np.isscalar(v) for v in info._asdict().values()])

    # Verify metrics
    assert 'approx_kl' in info._asdict()
    assert 'clipfrac' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])


@pytest.mark.unittest
def test_rloo_policy_loss_with_mask(batch_size, seq_length, dictionary_num):
    """Test RLOO policy loss calculation with mask"""
    # Create test data
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1
    action = torch.randint(0, dictionary_num, (batch_size, seq_length))
    reward = torch.randn(batch_size)
    action_mask = torch.ones(batch_size, seq_length)
    action_mask[:, -2:] = 0

    # Calculate loss
    data = rloo_policy_data(logit_new=logit_new, logit_old=logit_old, action=action, reward=reward, weight=action_mask)
    loss, info = rloo_policy_error(data, clip_ratio=0.2)

    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Ensure scalar output
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert logit_new.grad is None
    loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    # Verify metrics
    assert 'approx_kl' in info._asdict()
    assert 'clipfrac' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])
