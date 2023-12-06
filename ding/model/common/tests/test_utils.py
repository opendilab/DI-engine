import pytest
import torch
from ding.model.common.utils import top_p_logits


@pytest.mark.unittest
class TestUtils:

    def test_top_p_logits(self):
        test_logit = torch.Tensor([[0., 0.91, 0.05, 0.04], [0.04, 0.46, 0.46, 0.04]])

        gt_logit = torch.Tensor([[0., 1., 0., 0.], [0., 0.5, 0.5, 0.]])

        pred_logit = top_p_logits(test_logit)
        assert torch.sum((gt_logit - pred_logit) ** 2).item() < 1e-8
