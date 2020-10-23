import os
import time

import pytest
import torch
import torch.nn as nn

from nervex.torch_utils.checkpoint_helper import CheckpointHelper


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
        path = 'model.pt'
        os.popen('rm -rf ' + path)
        time.sleep(1)
        dst_model = DstModel()
        src_model = SrcModel()
        ckpt_state_dict = {'model': src_model.state_dict()}
        torch.save(ckpt_state_dict, path)

        ckpt_helper = CheckpointHelper()
        with pytest.raises(RuntimeError):
            ckpt_helper.load(path, dst_model, strict=True)

        ckpt_helper.load(path, dst_model, strict=False)
        assert torch.abs(dst_model.fc1.weight - src_model.fc1.weight).max() < 1e-6
        assert torch.abs(dst_model.fc1.bias - src_model.fc1.bias).max() < 1e-6
        os.popen('rm -rf ' + path)
