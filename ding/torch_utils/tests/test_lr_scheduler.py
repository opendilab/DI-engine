import pytest
import torch
from torch.optim import Adam

from ding.torch_utils.lr_scheduler import cos_lr_scheduler


@pytest.mark.unittest
class TestLRSchedulerHelper:

    def test_cos_lr_scheduler(self):
        r"""
        Overview:
            Test the cos lr scheduler.
        """
        net = torch.nn.Linear(3, 4)
        opt = Adam(net.parameters(), lr=1e-2)
        scheduler = cos_lr_scheduler(opt, learning_rate=1e-2, min_lr=6e-5)
        scheduler.step(101)
        assert opt.param_groups[0]['lr'] == 6e-5
