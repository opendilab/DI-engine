import os
import time

import pytest
import torch
from easydict import EasyDict
from typing import Any
from functools import partial

from nervex.utils import read_config
from nervex.worker import BaseLearner
from nervex.worker.learner import LearnerHook, register_learner_hook, add_learner_hook, \
    register_learner, create_learner


class FakeLearner(BaseLearner):
    @staticmethod
    def random_data():
        return {
            'obs': torch.randn(2),
            'replay_buffer_idx': 0,
            'replay_unique_id': 0,
        }

    def get_data(self, batch_size):
        return [self.random_data for _ in range(batch_size)]


class FakePolicy:

    def __init__(self):
        self._model = torch.nn.Identity()

    def forward(self, x):
        return {'total_loss': torch.randn(1).squeeze(), 'cur_lr': 0.1}

    def data_preprocess(self, x):
        return x

    def state_dict_handle(self):
        return {'model': self._model}

    def info(self):
        return 'FakePolicy'

    def monitor_vars(self):
        return ['total_loss', 'cur_lr']


@pytest.mark.unittest
class TestBaseLearner:

    def test_naive(self):
        os.popen('rm -rf ckpt*')
        os.popen('rm -rf iteration_5.pth.tar*')
        time.sleep(1.0)
        register_learner('fake', FakeLearner)
        path = os.path.join(os.path.dirname(__file__), './iteration_5.pth.tar')
        torch.save({'model': {}, 'last_iter': 5}, path)
        time.sleep(0.5)
        cfg = read_config(os.path.join(os.path.dirname(__file__), 'test_learner.yaml'))
        cfg.common.load_path = path
        learner = create_learner(EasyDict(cfg))
        learner.policy = FakePolicy()
        learner.launch()
        with pytest.raises(KeyError):
            create_learner(EasyDict({'learner': {'learner_type': 'placeholder', 'import_names': []}}))
        learner.run()
        time.sleep(2)
        assert learner.last_iter.val == 10 + 5

        # test hook
        dir_name = 'ckpt_{}'.format(learner.name)
        for n in [5, 10, 15]:
            assert os.path.exists(dir_name + '/iteration_{}.pth.tar'.format(n))
        for n in [0, 4, 7, 12]:
            assert not os.path.exists(dir_name + '/iteration_{}.pth.tar'.format(n))

        class FakeHook(LearnerHook):

            def __call__(self, engine: Any) -> Any:
                pass

        original_hook_num = len(learner._hooks['after_run'])
        add_learner_hook(learner._hooks, FakeHook(name='fake_hook', priority=30, position='after_run'))
        assert len(learner._hooks['after_run']) == original_hook_num + 1

        os.popen('rm -rf iteration_5.pth.tar*')
        os.popen('rm -rf ' + dir_name)
        os.popen('rm -rf learner')


@pytest.mark.unittest
class TestLearnerHook:

    def test_register(self):

        class FakeHook(LearnerHook):
            pass

        register_learner_hook('fake', FakeHook)
        with pytest.raises(AssertionError):
            register_learner_hook('placeholder', type)
