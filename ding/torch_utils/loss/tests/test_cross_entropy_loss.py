import pytest
import torch
import torch.nn as nn

from ding.torch_utils import LabelSmoothCELoss, SoftFocalLoss


@pytest.mark.unittest
class TestLabelSmoothCE:

    def test_label_smooth_ce_loss(self):
        logits = torch.randn(4, 6)
        labels = torch.LongTensor([i for i in range(4)])
        criterion1 = LabelSmoothCELoss(0)
        criterion2 = nn.CrossEntropyLoss()
        assert (torch.abs(criterion1(logits, labels) - criterion2(logits, labels)) < 1e-6)


@pytest.mark.unittest
class TestSoftFocalLoss:

    def test_soft_focal_loss(self):
        logits = torch.randn(4, 6)
        labels = torch.LongTensor([i for i in range(4)])
        criterion = SoftFocalLoss()
        loss = criterion(logits, labels)
        assert loss.shape == ()
        loss_value = loss.item()
