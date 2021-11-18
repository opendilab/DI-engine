import pytest
import torch
from ding.torch_utils.network import resnet18


@pytest.mark.unittest
def test_resnet18():
    model = resnet18()
    print(model)
    inputs = torch.randn(4, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (4, 1000)
