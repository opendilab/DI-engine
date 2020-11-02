import time

import pytest
import torch

from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager


@pytest.mark.unittest
class TestBaseEnvManager:

    def test_naive(self, setup_sync_manager_cfg):
        env_manager = BaseEnvManager(**setup_sync_manager_cfg)
        obs = env_manager.reset(reset_param=[{'stat': 'stat_test'} for _ in range(env_manager.env_num)])
        assert all([s == 'stat_test'] for s in env_manager._stat)
        env_manager.seed([314 for _ in range(env_manager.env_num)])
        assert all([s == 314 for s in env_manager._seed])
        timestep = env_manager.step([torch.randn(4) for _ in range(env_manager.env_num)])
        print(timestep)
        count = 1
        start_time = time.time()
        while not env_manager.all_done:
            env_id = [i for i, d in env_manager.env_done.items() if not d]
            action = [torch.randn(4) for _ in range(len(env_id))]
            timestep = env_manager.step(action, env_id)
            assert len(timestep) == len(env_id)
            print(timestep, count)
            count += 1
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))
        assert all(env_manager.env_done.values())
        assert env_manager._current_step[1] != 0
        env_manager.reset(reset_param=[{'stat': 'stat_test1'}], env_id=[1])
        assert env_manager.env_done[1] is False
        assert env_manager._current_step[1] == 0
        assert env_manager._stat[1] == 'stat_test1'
        assert all([env_manager._current_step[i] != 0 for i in [0, 2, 3]])

        env_manager.close()
