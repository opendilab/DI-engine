import time

import pytest
import torch

from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager, SyncSubprocessEnvManager


class TestBaseEnvManager:

    def test_naive(self, setup_async_manager_cfg, setup_model_type):
        env_manager = SubprocessEnvManager(**setup_async_manager_cfg)
        model = setup_model_type()

        env_manager.seed([314 for _ in range(env_manager.env_num)])
        assert all([s == 314 for s in env_manager._seed])
        env_manager.launch(reset_param=[{'stat': 'stat_test'} for _ in range(env_manager.env_num)])
        assert all([s == 'stat_test'] for s in env_manager._stat)

        env_count = [0 for _ in range(env_manager.env_num)]
        data_count = 0
        start_time = time.time()
        while not env_manager.done:
            obs = env_manager.next_obs
            print('obs', obs.keys(), env_manager._env_state)
            action = model.forward(obs)
            assert 1 <= len(action) <= len(obs)
            print('act', action.keys())
            timestep = env_manager.step(action)
            data_count += len(timestep)
            assert len(timestep) >= 1
            print('timestep', timestep.keys(), timestep)
            for k, t in timestep.items():
                if t.done:
                    print('env{} finish episode{}'.format(k, env_count[k]))
                    env_count[k] += 1
        assert all([c == setup_async_manager_cfg.episode_num for c in env_count])
        assert data_count == sum(env_manager._data_count)
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))

        env_manager.close()

    @pytest.mark.unittest
    def test_error(self, setup_async_manager_cfg, setup_exception):
        env_manager = SyncSubprocessEnvManager(**setup_async_manager_cfg)
        obs = env_manager.launch(reset_param=[{'stat': 'stat_test'} for _ in range(env_manager.env_num)])
        with pytest.raises(AttributeError):
            data = env_manager.xxx
        with pytest.raises(AttributeError):
            data = env_manager.xxx()
        timestep = env_manager.step({i: torch.randn(4) for i in range(env_manager.env_num)})
        assert len(timestep) == env_manager.env_num
        with pytest.raises(TypeError):
            env_manager.info()
        name = env_manager._name
        assert len(name) == env_manager.env_num
        assert all([isinstance(n, str) for n in name])
        name = env_manager.name
        assert len(name) == env_manager.env_num
        assert all([isinstance(n, str) for n in name])
        with pytest.raises(setup_exception):
            timestep = env_manager.step({i: 'error' for i in range(env_manager.env_num)})
        assert env_manager._closed

        env_manager.close()
        with pytest.raises(AssertionError):
            env_manager.reset([])
        with pytest.raises(AssertionError):
            env_manager.step([])
