import os
import time

import pytest
import torch
from easydict import EasyDict
from typing import Any

from nervex.worker import BaseLearner
from nervex.worker.learner import LearnerHook, register_learner_hook, add_learner_hook, \
    register_learner, create_learner


class FakeLearner(BaseLearner):

    def _setup_data_source(self):

        class DataLoader:

            def __next__(self):
                return torch.randn(4, 2)

        self._data_source = DataLoader()

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
        setattr(self._agent.model, 'load_state_dict', lambda x, strict: 0)


@pytest.mark.unittest
class TestBaseLearner:

    def test_naive(self):
        os.popen('rm -rf ckpt')
        os.popen('rm -rf iteration_5.pth.tar*')
        time.sleep(1.0)
        register_learner('fake', FakeLearner)
        path = os.path.join(os.path.dirname(__file__), './iteration_5.pth.tar')
        torch.save({'model': {}, 'last_iter': 5}, path)
        time.sleep(0.5)
        cfg = {
            'common': {
                'load_path': path
            },
            'learner': {
                'learner_type': 'fake',
                'import_names': []
            }
        }
        learner = create_learner(EasyDict(cfg))
        with pytest.raises(KeyError):
            create_learner(EasyDict({'learner': {'learner_type': 'placeholder', 'import_names': []}}))
        learner.run()
        time.sleep(2)
        assert learner.last_iter.val == 10 + 5

        # test hook
        assert learner.log_buffer == {}
        for n in [5, 10, 15]:
            assert os.path.exists('ckpt/iteration_{}.pth.tar'.format(n))
        for n in [0, 4, 7, 12]:
            assert not os.path.exists('ckpt/iteration_{}.pth.tar'.format(n))

        class FakeHook(LearnerHook):

            def __call__(self, engine: Any) -> Any:
                pass

        original_hook_num = len(learner._hooks['after_run'])
        add_learner_hook(learner._hooks, FakeHook(name='fake_hook', priority=30, position='after_run'))
        assert len(learner._hooks['after_run']) == original_hook_num + 1

        os.popen('rm -rf iteration_5.pth.tar*')
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
