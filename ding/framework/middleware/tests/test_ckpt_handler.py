import pytest

from easydict import EasyDict
from ding.framework import OnlineRLContext
from ding.framework.middleware.ckpt_handler import CkptSaver

import torch.nn as nn
import torch.optim as optim
import os
import shutil

from unittest.mock import Mock
from ding.framework import task

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MockModel(Mock):
    def __init__(self) -> None:
        super(MockModel, self).__init__()
    def state_dict(self):
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return model.state_dict()

class MockPolicy(Mock):
    def __init__(self) -> None:
        super(MockPolicy, self).__init__()
        self.learn_mode = MockModel()


@pytest.mark.unittest
def test_CkptSaver():
    cfg = EasyDict({'exp_name': 'test_CkptSaver_exp'})

    ctx = OnlineRLContext()
    
    train_freq = 100
    policy = MockPolicy()

    if not os.path.exists(cfg.exp_name):
        os.mkdir(cfg.exp_name)
    
    prefix = '{}/ckpt'.format(cfg.exp_name)

    with task.start():
        ctx.train_iter = 0
        ctx.eval_value = 9.4
        ckpt_saver = CkptSaver(cfg,policy,train_freq)
        ckpt_saver.__call__(ctx)
        assert os.path.exists("{}/eval.pth.tar".format(prefix))  

        ctx.train_iter = 100
        ctx.eval_value = 1
        ckpt_saver.__call__(ctx)
        assert os.path.exists("{}/iteration_{}.pth.tar".format(prefix, ctx.train_iter))

        task.finish = True
        ckpt_saver.__call__(ctx)
        assert os.path.exists("{}/final.pth.tar".format(prefix))

    shutil.rmtree(cfg.exp_name)