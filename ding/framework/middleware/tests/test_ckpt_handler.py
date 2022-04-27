import pytest

from easydict import EasyDict
from ding.framework import OnlineRLContext
from ding.framework.middleware.ckpt_handler import CkptSaver

import torch.nn as nn
import torch.optim as optim
import os
import shutil

from unittest.mock import Mock, patch
from ding.framework import task
from ding.utils import save_file


class TheModelClass(nn.Module):

    def state_dict(self):
        return 'fake_state_dict'


class MockPolicy(Mock):

    def __init__(self, model) -> None:
        super(MockPolicy, self).__init__()
        self.learn_mode = model


@pytest.mark.unittest
def test_ckpt_saver():
    cfg = EasyDict({'exp_name': 'test_ckpt_saver_exp'})

    ctx = OnlineRLContext()

    train_freq = 100
    model = TheModelClass()

    if not os.path.exists(cfg.exp_name):
        os.mkdir(cfg.exp_name)

    prefix = '{}/ckpt'.format(cfg.exp_name)

    with patch("ding.policy.Policy", MockPolicy), task.start():
        policy = MockPolicy(model)

        def mock_save_file(path, data, fs_type=None, use_lock=False):
            assert path == "{}/eval.pth.tar".format(prefix)

        with patch("ding.framework.middleware.ckpt_handler.save_file", mock_save_file):
            ctx.train_iter = 0
            ctx.eval_value = 9.4
            ckpt_saver = CkptSaver(cfg, policy, train_freq)
            ckpt_saver(ctx)

        def mock_save_file(path, data, fs_type=None, use_lock=False):
            assert path == "{}/iteration_{}.pth.tar".format(prefix, ctx.train_iter)

        with patch("ding.framework.middleware.ckpt_handler.save_file", mock_save_file):
            ctx.train_iter = 100
            ctx.eval_value = 1
            ckpt_saver(ctx)

        def mock_save_file(path, data, fs_type=None, use_lock=False):
            assert path == "{}/final.pth.tar".format(prefix)

        with patch("ding.framework.middleware.ckpt_handler.save_file", mock_save_file):
            task.finish = True
            ckpt_saver(ctx)

    shutil.rmtree(cfg.exp_name)
