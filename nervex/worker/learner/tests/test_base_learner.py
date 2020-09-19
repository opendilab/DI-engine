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
                return torch.randn(4, 2)

        self._data_source = DataLoader()

    def _setup_optimizer(self):
        self._optimizer = torch.optim.Adam(self._agent.model.parameters(), 0.1)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[5], gamma=0.1)

    def _setup_computation_graph(self):
        class Graph:
            def forward(self, data, agent):
                return {
                    'total_loss': agent.model(data).mean(),
                }

            def register_stats(self, record, tb_logger):
                record.register_var('total_loss')

                tb_logger.register_var('total_loss')

            def __repr__(self):
                return 'FakeComputationGraph'

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        self._computation_graph = Graph()

    def _setup_agent(self):
        class Agent():
            def __repr__(self):
                return 'FakeAgent'

        self._agent = Agent()
        setattr(self._agent, 'model', torch.nn.Linear(2, 2))


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
        os.popen('rm -rf default_*')


@pytest.mark.unittest
class TestLearnerHook:
    def test_register(self):
        class FakeHook(LearnerHook):
            pass

        register_learner_hook('fake', FakeHook)
        with pytest.raises(AssertionError):
            register_learner_hook('placeholder', type)
