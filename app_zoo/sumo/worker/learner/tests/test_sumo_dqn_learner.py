import os
import time
from threading import Thread

import pytest

from app_zoo.sumo.envs import FakeSumoDataset
from app_zoo.sumo.worker.learner.sumo_dqn_learner import SumoDqnLearner

# @pytest.mark.unittest
# class TestSumoDqnLearner:

#     def test_data_sample_update(self):
#         sumo_learner = SumoDqnLearner({})
#         dataset = FakeSumoDataset()
#         sumo_learner.get_data = lambda x: dataset.get_batch_sample(x)
#         sumo_learner.launch()
#         sumo_learner.run()

#     def clean_dist_train(self, name=''):
#         os.popen("rm -rf default*")
#         os.popen("rm -rf data")
#         os.popen("rm -rf ckpt_" + name)
#         os.popen("rm -rf api-log")
#         os.popen("rm -rf *.pth")
#         os.popen("rm -rf *.pth.lock")

#     def fake_push_data(self, coordinator, learner_uid):
#         time.sleep(3)  # monitor empty replay_buffer state
#         dataset = FakeSumoDataset(use_meta=True)
#         replay_buffer_handle = coordinator._learner_record[learner_uid]['replay_buffer']
#         for i in range(64):
#             replay_buffer_handle.push_data(dataset[i])
#         time.sleep(1)  # wait the cache flush out
#         assert (64 == replay_buffer_handle._meta_buffer.validlen)

#     def test_dist_train(self, setup_config, coordinator, league_manager):
#         self.clean_dist_train()
#         time.sleep(1)

#         learner = SumoDqnLearner(setup_config)
#         push_data_thread = Thread(target=self.fake_push_data, args=(coordinator, learner._learner_uid))
#         push_data_thread.daemon = True
#         push_data_thread.start()
#         learner.launch()
#         time.sleep(5)
#         learner.run(5)
#         assert learner.last_iter.val == 5
#         learner.close()

#         self.clean_dist_train(learner.name)
