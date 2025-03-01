import pytest
import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Tuple
from ding.rl_utils.log_prob_utils import (efficient_method, naive_method, less_efficient_method, LogProbFunction)


def get_gpu_memory() -> float:
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
    return 0


@pytest.fixture
def batch_size() -> int:
    return 16


@pytest.fixture
def seq_length() -> int:
    return 1024


@pytest.fixture
def dictionary_num() -> int:
    return 32768


@pytest.mark.gputest
def test_log_prob_methods_float32(batch_size: int, seq_length: int, dictionary_num: int) -> None:
    """Benchmark different methods for calculating log probabilities with float32"""
    print("\n" + "=" * 50)
    print("Testing with float32 precision")
    print("=" * 50)

    # 设置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计

    # 生成测试数据
    logits: Tensor = torch.randn(batch_size, seq_length, dictionary_num, device=device, dtype=torch.float32)
    input_ids: Tensor = torch.randint(0, dictionary_num, (batch_size, seq_length), device=device)

    # 预热 GPU
    for _ in range(3):
        _ = naive_method(logits[:2], input_ids[:2])
    torch.cuda.synchronize()

    # 测试每个方法
    results: Dict[str, Tensor] = {}
    peak_memory: Dict[str, float] = {}
    for method, name in [(naive_method, "Naive"), (efficient_method, "Efficient"),
                         (less_efficient_method, "Less_Efficient")]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # 重置每个方法的内存统计

        # 运行多次并计时
        times: List[float] = []
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

        # 记录内存使用
        peak_memory[name] = get_gpu_memory()

        # 计算统计信息
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n{name}:")
        print(f"Time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Peak GPU Memory: {peak_memory[name]:.2f} MB")

    # 验证结果正确性
    for name, result in results.items():
        if name != "Naive":
            diff = (results["Naive"] - result).abs().max().item()
            assert diff < 1e-5, f"Results mismatch between Naive and {name}: {diff}"


@pytest.mark.gputest
def test_log_prob_methods_bfloat16(batch_size: int, seq_length: int, dictionary_num: int) -> None:
    """Benchmark different methods for calculating log probabilities with bfloat16"""
    print("\n" + "=" * 50)
    print("Testing with bfloat16 precision")
    print("=" * 50)

    # 设置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tolerance = 0.1  # bfloat16的容差值要更大一些

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计

    # 生成测试数据
    logits: Tensor = torch.randn(batch_size, seq_length, dictionary_num, device=device, dtype=torch.bfloat16)
    input_ids: Tensor = torch.randint(0, dictionary_num, (batch_size, seq_length), device=device)

    # 预热 GPU
    for _ in range(3):
        _ = naive_method(logits[:2], input_ids[:2])
    torch.cuda.synchronize()

    # 测试每个方法
    results: Dict[str, Tensor] = {}
    peak_memory: Dict[str, float] = {}
    for method, name in [(naive_method, "Naive"), (efficient_method, "Efficient"),
                         (less_efficient_method, "Less_Efficient")]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # 重置每个方法的内存统计

        # 运行多次并计时
        times: List[float] = []
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

        # 记录内存使用
        peak_memory[name] = get_gpu_memory()

        # 计算统计信息
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n{name}:")
        print(f"Time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Peak GPU Memory: {peak_memory[name]:.2f} MB")

    # 验证结果正确性
    for name, result in results.items():
        if name != "Naive":
            diff = (results["Naive"] - result).abs().max().item()
            assert diff < tolerance, f"Results mismatch between Naive and {name}: {diff}"
