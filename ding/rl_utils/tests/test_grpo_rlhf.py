import pytest
import numpy as np
import torch
from ding.rl_utils.grpo import (
    grpo_policy_data, grpo_policy_error, naive_method, efficient_method, less_efficient_method
)


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
    loss, info = grpo_policy_error(data=data)

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
    assert 'mean_kl' in info._asdict()
    assert 'mean_ratio' in info._asdict()
    assert 'mean_clipped' in info._asdict()
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
    assert 'mean_kl' in info._asdict()
    assert 'mean_ratio' in info._asdict()
    assert 'mean_clipped' in info._asdict()
    assert all([np.isscalar(v) for v in info._asdict().values()])


@pytest.mark.benchmark
def test_log_prob_methods_benchmark():
    """Benchmark different methods for calculating log probabilities"""
    # 设置参数
    vocab_size = 32768
    seq_len = 1024
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成测试数据
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # 预热 GPU
    for _ in range(3):
        _ = naive_method(logits[:2], input_ids[:2])
    torch.cuda.synchronize()

    # 测试每个方法
    results = {}
    for method, name in [(naive_method, "Naive"), (efficient_method, "Efficient"),
                         (less_efficient_method, "Less_Efficient")]:
        # 运行多次并计时
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = method(logits, input_ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            if len(times) == 1:
                results[name] = result

        # 计算统计信息
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n{name}: {mean_time:.2f} ± {std_time:.2f} ms")

    # 验证结果正确性
    for name, result in results.items():
        if name != "Naive":
            diff = (results["Naive"] - result).abs().max().item()
            assert diff < 1e-5, f"Results mismatch between Naive and {name}: {diff}"
