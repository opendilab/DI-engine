import os
import time

import pytest
import torch
from easydict import EasyDict
from typing import Any
from functools import partial

from ding.worker import BaseLearner
from ding.worker.learner import LearnerHook, add_learner_hook, create_learner


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
        return {
            'total_loss': torch.randn(1).squeeze(),
            'cur_lr': 0.1,
            'priority': [1., 2., 3.],
            '[histogram]h_example': [1.2, 2.3, 3.4],
            '[scalars]s_example': {
                'a': 5.,
                'b': 4.
            },
        }

    def data_preprocess(self, x):
        return x

    def state_dict(self):
        return {'model': self._model}

    def load_state_dict(self, state_dict):
        pass

    def info(self):
        return 'FakePolicy'

    def monitor_vars(self):
        return ['total_loss', 'cur_lr']

    def get_attribute(self, name):
        if name == 'cuda':
            return False
        elif name == 'device':
            return 'cpu'
        elif name == 'batch_size':
            return 2
        elif name == 'on_policy':
            return False
        else:
            raise KeyError

    def reset(self):
        pass


@pytest.mark.unittest
class TestBaseLearner:

    def _get_cfg(self, path):
        cfg = BaseLearner.default_config()
        cfg.import_names = []
        cfg.learner_type = 'fake'
        cfg.train_iterations = 10
        cfg.hook.load_ckpt_before_run = path
        cfg.hook.log_show_after_iter = 5
        # Another way to build hook: Complete config
        cfg.hook.save_ckpt_after_iter = dict(
            name='save_ckpt_after_iter', type='save_ckpt', priority=40, position='after_iter', ext_args={'freq': 5}
        )

        return cfg

    def test_naive(self):
        os.popen('rm -rf iteration_5.pth.tar*')
        time.sleep(1.0)
        with pytest.raises(KeyError):
            create_learner(EasyDict({'type': 'placeholder', 'import_names': []}))
        path = os.path.join(os.path.dirname(__file__), './iteration_5.pth.tar')
        torch.save({'model': {}, 'last_iter': 5}, path)
        time.sleep(0.5)
        cfg = self._get_cfg(path)
        learner = FakeLearner(cfg, exp_name='exp_test')
        learner.policy = FakePolicy()
        learner.setup_dataloader()
        learner.start()
        time.sleep(2)
        assert learner.last_iter.val == 10 + 5

        # test hook
        dir_name = '{}/ckpt'.format(learner.exp_name)
        for n in [5, 10, 15]:
            assert os.path.exists(dir_name + '/iteration_{}.pth.tar'.format(n))
        for n in [0, 4, 7, 12]:
            assert not os.path.exists(dir_name + '/iteration_{}.pth.tar'.format(n))
        learner.debug('iter [5, 10, 15] exists; iter [0, 4, 7, 12] does not exist.')

        learner.save_checkpoint('best')

        info = learner.learn_info
        for info_name in ['learner_step', 'priority_info', 'learner_done']:
            assert info_name in info

        class FakeHook(LearnerHook):

            def __call__(self, engine: Any) -> Any:
                pass

        original_hook_num = len(learner._hooks['after_run'])
        add_learner_hook(learner._hooks, FakeHook(name='fake_hook', priority=30, position='after_run'))
        assert len(learner._hooks['after_run']) == original_hook_num + 1

        os.popen('rm -rf iteration_5.pth.tar*')
        os.popen('rm -rf ' + dir_name)
        os.popen('rm -rf learner')
        os.popen('rm -rf log')
        learner.close()
