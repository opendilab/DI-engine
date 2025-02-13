import pytest
import numpy as np
import torch
from ding.rl_utils.rloo import rloo_policy_data, rloo_policy_error


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
def test_rloo_policy_loss_without_mask(batch_size: int, seq_length: int, dictionary_num: int):
    """测试不带掩码的RLOO策略损失计算"""
    # 创建测试数据
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1  # 稍微偏离当前策略
    action = torch.randint(0, dictionary_num, (batch_size, seq_length))
    advantages = torch.randn(batch_size)  # RLOO中每个序列只有一个优势值

    # 计算损失
    data = rloo_policy_data(
        logit_new=logit_new,
        logit_old=logit_old,
        action=action,
        adv=advantages,
        weight=None
    )
    loss, info = rloo_policy_error(data, clip_ratio=0.2)

    # 验证输出
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # 标量
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert all([np.isscalar(v) for v in info._asdict().values()])


@pytest.mark.unittest
def test_rloo_policy_loss_with_mask(batch_size: int, seq_length: int, dictionary_num: int):
    """测试带掩码的RLOO策略损失计算"""
    # 创建测试数据
    logit_new = torch.randn(batch_size, seq_length, dictionary_num).requires_grad_(True)
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1  # 稍微偏离当前策略
    action = torch.randint(0, dictionary_num, (batch_size, seq_length))
    advantages = torch.randn(batch_size)  # RLOO中每个序列只有一个优势值
    action_mask = torch.ones(batch_size, seq_length)
    action_mask[:, -2:] = 0  # 设置最后两个时间步为padding

    # 计算损失
    data = rloo_policy_data(
        logit_new=logit_new,
        logit_old=logit_old,
        action=action,
        adv=advantages,
        weight=action_mask
    )
    loss, info = rloo_policy_error(data, clip_ratio=0.2)

    # 验证输出
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # 标量
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    # 验证指标
    assert 'mean_ratio' in info._asdict()
    assert 'mean_clipped' in info._asdict()
    assert 'mean_advantage' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])


