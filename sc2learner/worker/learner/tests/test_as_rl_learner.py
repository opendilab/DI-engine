import pytest
import os
import torch
import time
from threading import Thread
from sc2learner.data import FakeActorDataset


@pytest.mark.unittest
class TestASRLLearner:
    def fake_push_data(self, coordinator, learner_uid):
        time.sleep(3)  # monitor empty replay_buffer state
        dataset = FakeActorDataset(use_meta=True)
        replay_buffer_handle = coordinator.learner_record[learner_uid]['replay_buffer']
        for i in range(64):
            replay_buffer_handle.push_data(dataset[i])
        time.sleep(1)  # wait the cache flush out
        assert (64 == replay_buffer_handle._meta_buffer.validlen)

    def test_data_sample_update(self, setup_config_real, coordinator, league_manager, real_learner):
        """
        Note: coordinator must be in the front of learner in the arguments
        """
        push_data_thread = Thread(target=self.fake_push_data, args=(coordinator, real_learner.learner_uid))
        push_data_thread.start()
        time.sleep(60)
