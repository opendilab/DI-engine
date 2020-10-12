import os

import pytest

from nervex.envs.sumo.fake_dataset import FakeSumoDataset
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner


@pytest.mark.unittest
class TestSumoDqnLearner:

    def test_data_sample_update(self):
        os.popen('rm -rf ckpt')
        sumo_learner = SumoDqnLearner({})
        dataset = FakeSumoDataset()
        sumo_learner.get_data = lambda x: dataset.get_batch_sample(x)
        sumo_learner.run()
        os.popen('rm -rf ckpt')
        os.popen('rm -rf default_*')
