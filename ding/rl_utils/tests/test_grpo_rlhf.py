import pytest
import numpy as np
import torch
from ding.rl_utils.grpo import grpo_policy_data, grpo_policy_error  # 导入GRPO相关函数


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
    """测试GRPO策略损失的计算"""
    # 1. 创建测试数据
    logit_new = torch.randn(batch_size, seq_length, vocab_size).requires_grad_(True)  # 当前策略的logits
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1  # 旧策略的logits（稍微偏离当前策略）
    logit_ref = logit_new + torch.randn_like(logit_new) * 0.2  # 参考策略的logits
    action = torch.randint(0, vocab_size, (batch_size, seq_length))  # 随机采样的token
    adv = torch.randn(batch_size)  # 每个序列的优势值
    weight = torch.ones(batch_size, seq_length)  # 掩码
    weight[:, -2:] = 0  # 设置最后两个时间步为padding

    # 2. 创建grpo_policy_data实例
    data = grpo_policy_data(
        logit_new=logit_new,  # 当前策略的输出
        logit_old=logit_old,  # 旧策略的输出
        logit_ref=logit_ref,  # 参考策略的输出
        action=action,  # 实际采样的token
        adv=adv,  # 优势值
        weight=weight  # 掩码
    )

    # 3. 计算GRPO损失
    loss, info = grpo_policy_error(
        data=data,
        clip_ratio=0.2,  # PPO截断比率
        beta=0.1  # KL散度权重
    )

    # 4. 验证输出
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # 确保是标量
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)

    # 5. 测试梯度
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    # 6. 验证指标
    assert 'mean_kl' in info._asdict()
    assert 'mean_ratio' in info._asdict()
    assert 'mean_clipped' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])


@pytest.mark.unittest
def test_grpo_policy_loss_without_mask(batch_size: int = 4, seq_length: int = 8, vocab_size: int = 1000):
    """测试GRPO策略损失的计算"""
    # 1. 创建测试数据
    logit_new = torch.randn(batch_size, seq_length, vocab_size).requires_grad_(True)  # 当前策略的logits
    logit_old = logit_new + torch.randn_like(logit_new) * 0.1  # 旧策略的logits（稍微偏离当前策略）
    logit_ref = logit_new + torch.randn_like(logit_new) * 0.2  # 参考策略的logits
    action = torch.randint(0, vocab_size, (batch_size, seq_length))  # 随机采样的token
    adv = torch.randn(batch_size)  # 每个序列的优势值


    # 2. 创建grpo_policy_data实例
    data = grpo_policy_data(
        logit_new=logit_new,  # 当前策略的输出
        logit_old=logit_old,  # 旧策略的输出
        logit_ref=logit_ref,  # 参考策略的输出
        action=action,  # 实际采样的token
        adv=adv,  # 优势值
        weight=None  # 掩码
    )

    # 3. 计算GRPO损失
    loss, info = grpo_policy_error(
        data=data,
        clip_ratio=0.2,  # PPO截断比率
        beta=0.1  # KL散度权重
    )

    # 4. 验证输出
    assert isinstance(loss.policy_loss, torch.Tensor)
    assert loss.policy_loss.shape == torch.Size([])  # 确保是标量
    assert not torch.isnan(loss.policy_loss)
    assert not torch.isinf(loss.policy_loss)

    # 5. 测试梯度
    assert logit_new.grad is None
    loss.policy_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)

    # 6. 验证指标
    assert 'mean_kl' in info._asdict()
    assert 'mean_ratio' in info._asdict()
    assert 'mean_clipped' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])