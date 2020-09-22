import pytest
import torch
import yaml
import os
import random
from easydict import EasyDict
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.envs.sumo.fake_dataset import FakeSumoDataset
from nervex.data import default_collate


@pytest.mark.unittest
class TestSumoDqnLearner:
    def test_data_sample_update(self):
        def fake_data_source(bs):
            dataset = FakeSumoDataset(bs).getBatchSample()
            while True:
                yield default_collate(next(dataset))

        os.popen('rm -rf ckpt')
        sumo_learner = SumoDqnLearner({})
        sumo_learner._data_source = fake_data_source(sumo_learner._cfg.learner.batch_size)
        sumo_learner.run()
        os.popen('rm -rf ckpt')
        os.popen('rm -rf default_*')
