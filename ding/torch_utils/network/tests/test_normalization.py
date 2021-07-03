import pytest
import torch
from ding.torch_utils import build_normalization

num_features = 2
batch_size = 2
H, W = 2, 3


@pytest.mark.unittest
class TestNormalization:

    def validate(self, input, norm):
        output = norm(input)
        loss = output.mean()
        loss.backward()
        assert output.shape == input.shape
        assert isinstance(input.grad, torch.Tensor)

    def test(self):
        with pytest.raises(KeyError):
            norm = build_normalization('XXXN')
        input1d = torch.rand(batch_size, num_features).requires_grad_(True)
        input2d = torch.rand(batch_size, num_features, H, W).requires_grad_(True)

        norm_type = 'BN'
        norm = build_normalization(norm_type, dim=1)(num_features)
        self.validate(input1d, norm)

        norm = build_normalization(norm_type, dim=2)(num_features)
        self.validate(input2d, norm)

        norm_type = 'LN'
        norm = build_normalization(norm_type)(input1d.shape[1:])
        self.validate(input1d, norm)

        norm = build_normalization(norm_type)(input2d.shape[2:])
        self.validate(input2d, norm)

        norm_type = 'IN'
        norm = build_normalization(norm_type, dim=2)(num_features)
        self.validate(input2d, norm)
