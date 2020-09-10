import pytest
import random
import torch
import os
from nervex.worker import BaseLearner
from nervex.worker.learner import LearnerHook, register_learner_hook


class FakeLearner(BaseLearner):
    def _setup_data_source(self):
        class DataLoader():
            def __next__(self):
                return {}

        self._data_source = DataLoader()

    def _setup_optimizer(self):
        class Optimizer:
            def __init__(self):
                class Agent():
                    pass

                self.agent = Agent()
                setattr(self.agent, 'model', torch.nn.Linear(2, 2))
                self.optimizer = torch.optim.Adam(self.agent.model.parameters(), 0.1)

            def learn(self, data):
                return {
                    'total_loss': random.uniform(0, 1),
                    'forward_time': random.uniform(0.1, 0.2),
                    'backward_time': random.uniform(0.01, 0.05)
                }

            def register_stats(self, record, tb_logger):
                record.register_var('total_loss')
                record.register_var('forward_time')
                record.register_var('backward_time')

                tb_logger.register_var('total_loss')
                tb_logger.register_var('forward_time')
                tb_logger.register_var('backward_time')

            def __repr__(self):
                return 'Optimizer'

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        self._optimizer = Optimizer()
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer.optimizer, milestones=[5], gamma=0.1)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler


@pytest.mark.unittest
class TestBaseLearner:
    def test_naive(self):
        os.popen('rm -rf ckpt')
        learner = FakeLearner({})
        learner.run()
        for n in [0, 5, 10]:
            assert os.path.exists('ckpt/iteration_{}.pth.tar'.format(n))
        for n in [1, 4, 7]:
            assert not os.path.exists('ckpt/iteration_{}.pth.tar'.format(n))
        assert learner.last_iter.val == 10
        os.popen('rm -rf ckpt')


@pytest.mark.unittest
class TestLearnerHook:
    def test_register(self):
        class FakeHook(LearnerHook):
            pass

        register_learner_hook('fake', FakeHook)
        with pytest.raises(AssertionError):
            register_learner_hook('placeholder', type)
