import pytest
import torch
import torch.nn as nn
from sc2learner.torch_utils.checkpoint_helper import CheckpointHelper


class DstModel(nn.Module):
    def __init__(self):
        super(DstModel, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 8)
        self.fc_dst = nn.Linear(3, 6)


class SrcModel(nn.Module):
    def __init__(self):
        super(SrcModel, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 10)
        self.fc_src = nn.Linear(3, 7)


@pytest.mark.unittest
class TestCkptHelper:
    def test_load_model(self):
        dst_model = DstModel()
        src_model = SrcModel()
        ckpt_state_dict = {'state_dict': src_model.state_dict()}

        ckpt_helper = CheckpointHelper()
        with pytest.raises(RuntimeError):
            ckpt_helper.load(ckpt_state_dict, dst_model, need_torch_load=False, strict=True)

        ckpt_helper.load(ckpt_state_dict, dst_model, need_torch_load=False, strict=False)
        assert torch.abs(dst_model.fc1.weight - src_model.fc1.weight).max() < 1e-6
        assert torch.abs(dst_model.fc1.bias - src_model.fc1.bias).max() < 1e-6
