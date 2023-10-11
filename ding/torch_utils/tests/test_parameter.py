import unittest
import pytest
import torch
from ding.torch_utils.parameter import NonegativeParameter, TanhParameter


@pytest.mark.unittest
def test_nonegative_parameter():
    nonegative_parameter = NonegativeParameter(torch.tensor([2.0, 3.0]))
    assert torch.sum(torch.abs(nonegative_parameter() - torch.tensor([2.0, 3.0]))) == 0
    nonegative_parameter.set_data(torch.tensor(1))
    assert nonegative_parameter() == 1


@pytest.mark.unittest
def test_tanh_parameter():
    tanh_parameter = TanhParameter(torch.tensor([0.5, -0.2]))
    assert torch.isclose(tanh_parameter() - torch.tensor([0.5, -0.2]), torch.zeros(2), atol=1e-6).all()
    tanh_parameter.set_data(torch.tensor(0.3))
    assert tanh_parameter() == 0.3


if __name__ == "__main__":
    test_nonegative_parameter()
    test_tanh_parameter()
