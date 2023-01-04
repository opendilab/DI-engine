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
from ding.policy.base_policy import Policy


class TheModelClass(nn.Module):

    def state_dict(self):
        return 'fake_state_dict'


class MockPolicy(Mock):

    def __init__(self, model, **kwargs) -> None:
        super(MockPolicy, self).__init__(model)
        self.learn_mode = model

    @property
    def eval_mode(self):
        return EasyDict({"state_dict": lambda: {}})


@pytest.mark.unittest
def test_ckpt_saver():
    exp_name = 'test_ckpt_saver_exp'

    ctx = OnlineRLContext()

    train_freq = 100
    model = TheModelClass()

    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    prefix = '{}/ckpt'.format(exp_name)

    with patch("ding.policy.Policy", MockPolicy), task.start():
        policy = MockPolicy(model)

        def mock_save_file(path, data, fs_type=None, use_lock=False):
            assert path == "{}/eval.pth.tar".format(prefix)

        with patch("ding.framework.middleware.ckpt_handler.save_file", mock_save_file):
            ctx.train_iter = 1
            ctx.eval_value = 9.4
            ckpt_saver = CkptSaver(policy, exp_name, train_freq)
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

    shutil.rmtree(exp_name)
