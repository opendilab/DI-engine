import os

import pytest

from app_zoo.gfootball.envs.fake_dataset import FakeGfootballDataset
from app_zoo.gfootball.worker.learner.gfootball_iql_learner import GfootballIqlLearner


@pytest.mark.envtest
class TestGfootballIqlLearner:

    def test_data_sample_update(self):
        os.popen('rm -rf ckpt')
        outer_cfg = {'model': {'placeholder': 0}}
        football_learner = GfootballIqlLearner(outer_cfg)
        dataset = FakeGfootballDataset()
        football_learner.get_data = lambda x: dataset.get_batch_sample(x)
        football_learner.run()
        os.popen('rm -rf ckpt')
        os.popen('rm -rf default_*')
