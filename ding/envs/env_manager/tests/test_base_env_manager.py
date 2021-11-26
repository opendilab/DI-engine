import time
import signal
import pytest
import torch
import numpy as np

from ..base_env_manager import BaseEnvManager, EnvState


@pytest.mark.unittest
class TestBaseEnvManager:

    def test_naive(self, setup_base_manager_cfg):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_manager = BaseEnvManager(env_fn, setup_base_manager_cfg)
        env_manager.seed([314 for _ in range(env_manager.env_num)])
        assert env_manager._closed
        obs = env_manager.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        assert all([env_manager._env_states[env_id] == EnvState.RUN for env_id in range(env_manager.env_num)])
        # Test basic
        name = env_manager._name
        assert len(name) == env_manager.env_num
        assert all([isinstance(n, str) for n in name])
        assert env_manager._max_retry == 5
        assert env_manager._reset_timeout == 10
        assert all([s == 314 for s in env_manager._seed])
        assert all([s == 'stat_test'] for s in env_manager._stat)
        # Test arribute
        with pytest.raises(AttributeError):
            _ = env_manager.xxx
        with pytest.raises(RuntimeError):
            env_manager.user_defined()
        # Test step
        count = 1
        start_time = time.time()
        while not env_manager.done:
            env_id = env_manager.ready_obs.keys()
            action = {i: np.random.randn(4) for i in env_id}
            timestep = env_manager.step(action)
            assert len(timestep) == len(env_id)
            print('Count {}'.format(count))
            print([v.info for v in timestep.values()])
            print([v.done for v in timestep.values()])
            count += 1
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))
        assert all([env_manager._env_states[env_id] == EnvState.DONE for env_id in range(env_manager.env_num)])
        assert all([c == setup_base_manager_cfg.episode_num for c in env_manager._env_episode_count.values()])
        # Test close
        env_manager.close()
        assert env_manager._closed
        assert all([not env_manager._envs[env_id]._launched for env_id in range(env_manager.env_num)])
        assert all([env_manager._env_states[env_id] == EnvState.VOID for env_id in range(env_manager.env_num)])
        with pytest.raises(AssertionError):
            env_manager.reset([])
        with pytest.raises(AssertionError):
            env_manager.step([])

    def test_error(self, setup_base_manager_cfg):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_manager = BaseEnvManager(env_fn, setup_base_manager_cfg)
        # Test reset error
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'error'} for i in range(env_manager.env_num)}
            obs = env_manager.launch(reset_param=reset_param)
        assert env_manager._closed
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        obs = env_manager.launch(reset_param=reset_param)
        assert not env_manager._closed

        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})
        assert len(timestep) == env_manager.env_num
        # Test reset error once
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        assert env_manager._retry_type == 'reset'
        env_id_0 = env_manager.time_id[0]
        reset_param[0] = {'stat': 'error_once'}
        env_manager.reset(reset_param)
        env_manager.reset(reset_param)
        assert not env_manager._closed
        assert env_manager.time_id[0] == env_id_0
        env_manager._retry_type = 'renew'
        env_id_0 = env_manager.time_id[0]
        reset_param[0] = {'stat': 'error_once'}
        env_manager.reset(reset_param)
        assert not env_manager._closed
        assert env_manager.time_id[0] != env_id_0

        # Test step catched error
        action = {i: np.random.randn(4) for i in range(env_manager.env_num)}
        action[0] = 'catched_error'
        timestep = env_manager.step(action)
        assert timestep[0].info['abnormal']
        assert all(['abnormal' not in timestep[i].info for i in range(1, env_manager.env_num)])
        assert all([env_manager._env_states[i] == EnvState.RUN for i in range(env_manager.env_num)])
        assert len(env_manager.ready_obs) == 4
        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})
        # Test step error
        action[0] = 'error'
        with pytest.raises(RuntimeError):
            timestep = env_manager.step(action)
        assert env_manager._env_states[0] == EnvState.ERROR
        assert all([env_manager._env_states[i] == EnvState.RUN for i in range(1, env_manager.env_num)])
        obs = env_manager.reset(reset_param)
        assert all([env_manager._env_states[i] == EnvState.RUN for i in range(env_manager.env_num)])
        assert len(env_manager.ready_obs) == 4
        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})

        env_manager.close()

    @pytest.mark.timeout(60)
    def test_block(self, setup_base_manager_cfg):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg['max_retry'] = 1
        env_manager = BaseEnvManager(env_fn, setup_base_manager_cfg)
        assert env_manager._max_retry == 1
        # Test reset timeout
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'block'} for i in range(env_manager.env_num)}
            obs = env_manager.launch(reset_param=reset_param)
        assert env_manager._closed
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        reset_param[0]['stat'] = 'wait'

        obs = env_manager.launch(reset_param=reset_param)
        assert not env_manager._closed

        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})
        assert len(timestep) == env_manager.env_num
        # Test step timeout
        action = {i: np.random.randn(4) for i in range(env_manager.env_num)}
        action[0] = 'block'
        with pytest.raises(RuntimeError):
            timestep = env_manager.step(action)
        assert all([env_manager._env_states[i] == EnvState.RUN for i in range(1, env_manager.env_num)])

        obs = env_manager.reset(reset_param)
        action[0] = 'wait'
        timestep = env_manager.step(action)
        assert len(timestep) == env_manager.env_num

        env_manager.close()
