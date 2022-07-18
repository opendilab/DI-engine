import pytest
import torch
from ding.torch_utils.reshape_helper import fold_batch, unfold_batch, unsqueeze_repeat


@pytest.mark.unittest
def test_fold_unfold_batch():
    T, B, C, H, W = 10, 20, 3, 255, 255
    data = torch.randn(T, B, C, H, W)
    data, batch_dim = fold_batch(data, nonbatch_ndims=3)
    assert data.shape == (T * B, C, H, W) and batch_dim == (T, B)
    data = unfold_batch(data, batch_dim)
    assert data.shape == (T, B, C, H, W)

    T, B, N = 10, 20, 100
    data = torch.randn(T, B, N)
    data, batch_dim = fold_batch(data, nonbatch_ndims=1)
    assert data.shape == (T * B, N) and batch_dim == (T, B)
    data = unfold_batch(data, batch_dim)
    assert data.shape == (T, B, N)


@pytest.mark.unittest
def test_unsqueeze_repeat():
    T, B, C, H, W = 10, 20, 3, 255, 255
    repeat_times = 4
    data = torch.randn(T, B, C, H, W)
    ensembled_data = unsqueeze_repeat(data, repeat_times)
    assert ensembled_data.shape == (repeat_times, T, B, C, H, W)

    ensembled_data = unsqueeze_repeat(data, repeat_times, -1)
    assert ensembled_data.shape == (T, B, C, H, W, repeat_times)

    ensembled_data = unsqueeze_repeat(data, repeat_times, 2)
    assert ensembled_data.shape == (T, B, repeat_times, C, H, W)
