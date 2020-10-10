import pytest
import torch
import yaml
import os
import random
from easydict import EasyDict
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.envs.sumo.fake_dataset import FakeSumoDataset


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
