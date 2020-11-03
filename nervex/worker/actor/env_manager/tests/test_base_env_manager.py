import time

import pytest
import torch

from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager


@pytest.mark.unittest
class TestBaseEnvManager:

    def test_naive(self, setup_sync_manager_cfg):
        env_manager = BaseEnvManager(**setup_sync_manager_cfg)
        env_manager.seed([314 for _ in range(env_manager.env_num)])
        obs = env_manager.launch(reset_param=[{'stat': 'stat_test'} for _ in range(env_manager.env_num)])
        assert all([s == 314 for s in env_manager._seed])
        assert all([s == 'stat_test'] for s in env_manager._stat)
        count = 1
        start_time = time.time()
        while not env_manager.done:
            env_id = env_manager.next_obs.keys()
            action = {i: torch.randn(4) for i in env_id}
            timestep = env_manager.step(action)
            assert len(timestep) == len(env_id)
            print(timestep, count)
            count += 1
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))
        assert all(env_manager._env_done.values())
        assert all([c == setup_sync_manager_cfg.episode_num for c in env_manager._env_episode_count.values()])

        env_manager.close()
