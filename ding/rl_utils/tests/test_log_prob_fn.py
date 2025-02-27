import pytest
import numpy as np
import torch
from ding.rl_utils.log_prob_utils import (efficient_method, naive_method, less_efficient_method)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def seq_length():
    return 1024


@pytest.fixture
def dictionary_num():
    return 32768


@pytest.mark.gputest
def test_log_prob_methods_benchmark():
    """Benchmark different methods for calculating log probabilities"""
    # 设置参数

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成测试数据
    logits = torch.randn(batch_size, seq_length, dictionary_num, device=device, dtype=torch.float32)
    input_ids = torch.randint(0, dictionary_num, (batch_size, seq_length), device=device)

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
