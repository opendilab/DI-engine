import pytest
import torch
from ding.torch_utils.network import GatingType, SumMerge, VectorMerge


@pytest.mark.unittest
def test_SumMerge():
    input_shape = (3, 5)
    input = [torch.rand(input_shape).requires_grad_(True) for i in range(4)]
    sum_merge = SumMerge()

    output = sum_merge(input)
    assert output.shape == (3, 5)
    loss = output.mean()
    loss.backward()
    assert isinstance(input[0].grad, torch.Tensor)


@pytest.mark.unittest
def test_VectorMerge():
    input_sizes = {'in1': 3, 'in2': 16, 'in3': 27}
    output_size = 512
    input_dict = {}
    for k, v in input_sizes.items():
        input_dict[k] = torch.rand((64, v)).requires_grad_(True)

    vector_merge = VectorMerge(input_sizes, output_size, GatingType.NONE)
    output = vector_merge(input_dict)
    assert output.shape == (64, output_size)
    loss = output.mean()
    loss.backward()
    assert isinstance(input_dict['in1'].grad, torch.Tensor)

    vector_merge = VectorMerge(input_sizes, output_size, GatingType.GLOBAL)
    output = vector_merge(input_dict)
    assert output.shape == (64, output_size)
    loss = output.mean()
    loss.backward()
    assert isinstance(input_dict['in1'].grad, torch.Tensor)

    vector_merge = VectorMerge(input_sizes, output_size, GatingType.POINTWISE)
    output = vector_merge(input_dict)
    assert output.shape == (64, output_size)
    loss = output.mean()
    loss.backward()
    assert isinstance(input_dict['in1'].grad, torch.Tensor)
