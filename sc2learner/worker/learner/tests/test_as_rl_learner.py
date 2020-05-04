import pytest
import os
import torch
import time
from threading import Thread
from sc2learner.data import FakeActorDataset


@pytest.mark.unittest
class TestASRLLearner:
    def fake_push_data(self, coordinator):
        time.sleep(3)  # monitor empty replay_buffer state
        dataset = FakeActorDataset(use_meta=True)
        for i in range(64):
            coordinator.replay_buffer.push_data(dataset[i])
        time.sleep(1)  # wait the cache flush out
        assert (64 == coordinator.replay_buffer._meta_buffer.validlen)

    def test_data_sample_update(self, setup_config_real, coordinator, league_manager, real_learner):
        """
        Note: coordinator must be in the front of learner in the arguments
        """
        push_data_thread = Thread(target=self.fake_push_data, args=(coordinator, ))
        push_data_thread.start()
        time.sleep(60)
