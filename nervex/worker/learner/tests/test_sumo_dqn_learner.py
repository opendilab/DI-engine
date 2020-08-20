import pytest
import torch
import yaml
import os
import random
from easydict import EasyDict
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'test_sumo_default.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


@pytest.mark.unittest
class TestSumoDqnLearner:
    def test_data_sample_update(self, setup_config):
        sumo_learner = SumoDqnLearner(setup_config)
        sumo_learner.run()
