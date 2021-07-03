import pytest
import torch
from ding.torch_utils import MultiLogitsLoss


@pytest.mark.unittest
@pytest.mark.parametrize('criterion_type', ['cross_entropy', 'label_smooth_ce'])
def test_multi_logits_loss(criterion_type):
    logits = torch.randn(4, 8).requires_grad_(True)
    label = torch.LongTensor([0, 1, 3, 2])
    criterion = MultiLogitsLoss(criterion=criterion_type)
    loss = criterion(logits, label)
    assert loss.shape == ()
    assert logits.grad is None
    loss.backward()
    assert isinstance(logits, torch.Tensor)
