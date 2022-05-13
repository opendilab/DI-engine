import time
import signal
import pytest
import torch
import numpy as np

from ..base_env_manager import EnvState
from ..subprocess_env_manager import AsyncSubprocessEnvManager, SyncSubprocessEnvManager


class TestSubprocessEnvManager:

    @pytest.mark.unittest
    def test_naive(self, setup_async_manager_cfg, setup_model_type):
        env_fn = setup_async_manager_cfg.pop('env_fn')
        env_manager = AsyncSubprocessEnvManager(env_fn, setup_async_manager_cfg)
        model = setup_model_type()

        env_manager.seed([314 for _ in range(env_manager.env_num)])
        env_manager.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        assert all([s == 314 for s in env_manager._seed])
        assert all([s == 'stat_test'] for s in env_manager._stat)
        # Test basic
        name = env_manager._name
        for i in range(env_manager.env_num):
            assert name[i] == 'name{}'.format(i)
        assert len(name) == env_manager.env_num
        assert all([isinstance(n, str) for n in name])
        name = env_manager.name
        assert len(name) == env_manager.env_num
        assert all([isinstance(n, str) for n in name])
        assert env_manager._max_retry == 2
        assert env_manager._connect_timeout == 8
        assert env_manager._step_timeout == 5
        # Test arribute
        with pytest.raises(AttributeError):
            data = env_manager.xxx
        env_manager._env_ref.user_defined()
        with pytest.raises(RuntimeError):
            env_manager.user_defined()
        # Test step
        env_count = [0 for _ in range(env_manager.env_num)]
        data_count = 0
        start_time = time.time()
        while not env_manager.done:
            obs = env_manager.ready_obs
            print('obs', obs.keys(), env_manager._env_states)
            action = model.forward(obs)
            assert 1 <= len(action) <= len(obs)
            print('act', action.keys())
            timestep = env_manager.step(action)
            data_count += len(timestep)
            assert len(timestep) >= 1
            print('timestep', timestep.keys(), timestep, len(timestep))
            for k, t in timestep.items():
                if t.done:
                    print('env{} finish episode{}'.format(k, env_count[k]))
                    env_count[k] += 1
        assert all([c == setup_async_manager_cfg.episode_num for c in env_count])
        assert data_count == sum(env_manager._data_count)
        assert all([env_manager._env_states[env_id] == EnvState.DONE for env_id in range(env_manager.env_num)])
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))

        # Test close
        env_manager.close()
        assert env_manager._closed
        with pytest.raises(AssertionError):
            env_manager.reset([])
        with pytest.raises(AssertionError):
            env_manager.step([])

    @pytest.mark.unittest
    def test_error(self, setup_sync_manager_cfg):
        env_fn = setup_sync_manager_cfg.pop('env_fn')
        env_manager = SyncSubprocessEnvManager(env_fn, setup_sync_manager_cfg)
        # Test reset error
        with pytest.raises(AssertionError):
            env_manager.reset(reset_param={i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        with pytest.raises(RuntimeError):
            env_manager.launch(reset_param={i: {'stat': 'error'} for i in range(env_manager.env_num)})
        assert env_manager._closed
        time.sleep(0.5)  # necessary time interval
        env_manager.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        assert not env_manager._closed

        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})
        assert len(timestep) == env_manager.env_num

        # Test reset error once
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        assert env_manager._retry_type == 'reset'
        env_id_0 = env_manager.time_id[0]
        reset_param[0] = {'stat': 'error_once'}
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
        assert not env_manager._closed
        timestep = env_manager.step(action)
        assert not env_manager._closed

        assert timestep[0].info['abnormal']
        assert all(['abnormal' not in timestep[i].info for i in range(1, env_manager.env_num)])
        assert env_manager._env_states[0] == EnvState.ERROR
        assert len(env_manager.ready_obs) == 3
        # wait for reset
        env_manager.reset({0: {'stat': 'stat_test'}})
        while not len(env_manager.ready_obs) == env_manager.env_num:
            time.sleep(0.1)
        assert env_manager._env_states[0] == EnvState.RUN
        assert len(env_manager.ready_obs) == 4
        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})

        # # Test step error
        action[0] = 'error'
        with pytest.raises(RuntimeError):
            timestep = env_manager.step(action)
        assert env_manager._closed

        env_manager.close()
        with pytest.raises(AssertionError):  # Assert env manager is not closed
            env_manager.reset([])
        with pytest.raises(AssertionError):  # Assert env manager is not closed
            env_manager.step([])

    @pytest.mark.tmp  # gitlab ci and local test pass, github always fail
    @pytest.mark.timeout(100)
    def test_block(self, setup_async_manager_cfg, setup_model_type):
        env_fn = setup_async_manager_cfg.pop('env_fn')
        env_manager = AsyncSubprocessEnvManager(env_fn, setup_async_manager_cfg)
        model = setup_model_type()
        # Test connect timeout
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'block'} for i in range(env_manager.env_num)}
            obs = env_manager.launch(reset_param=reset_param)
        assert env_manager._closed
        time.sleep(0.5)
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        reset_param[0]['stat'] = 'wait'
        env_manager.launch(reset_param=reset_param)
        time.sleep(0.5)
        assert not env_manager._closed

        timestep = env_manager.step({i: np.random.randn(4) for i in range(env_manager.env_num)})
        obs = env_manager.ready_obs
        assert len(obs) >= 1

        # Test reset timeout
        env_manager._connect_timeout = 30
        env_manager._reset_timeout = 8
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'block'} for i in range(env_manager.env_num)}
            obs = env_manager.reset(reset_param=reset_param)
        assert env_manager._closed
        time.sleep(0.5)
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        reset_param[0]['stat'] = 'wait'
        env_manager.launch(reset_param=reset_param)
        time.sleep(0.5)
        assert not env_manager._closed

        # Test step timeout
        env_manager._step_timeout = 5
        obs = env_manager.reset({i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        action = {i: np.random.randn(4) for i in range(env_manager.env_num)}
        action[0] = 'block'
        with pytest.raises(TimeoutError):
            timestep = env_manager.step(action)
            obs = env_manager.ready_obs
            while 0 not in obs:
                action = model.forward(obs)
                timestep = env_manager.step(action)
                obs = env_manager.ready_obs
        time.sleep(0.5)

        obs = env_manager.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_manager.env_num)})
        time.sleep(1)
        action[0] = 'wait'
        timestep = env_manager.step(action)
        obs = env_manager.ready_obs
        while 0 not in obs:
            action = model.forward(obs)
            timestep = env_manager.step(action)
            obs = env_manager.ready_obs
        assert len(obs) >= 1

        env_manager.close()

    @pytest.mark.unittest
    def test_reset(self, setup_async_manager_cfg, setup_model_type):
        env_fn = setup_async_manager_cfg.pop('env_fn')
        setup_async_manager_cfg['auto_reset'] = False
        env_manager = AsyncSubprocessEnvManager(env_fn, setup_async_manager_cfg)
        model = setup_model_type()
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_manager.env_num)}
        obs = env_manager.launch(reset_param=reset_param)
        while True:
            obs = env_manager.ready_obs
            action = model.forward(obs)
            timestep = env_manager.step(action)
            if env_manager.done:
                break
            for env_id, t in timestep.items():
                if t.done and not env_manager.env_state_done(env_id):
                    env_manager.reset({env_id: None})
        assert all(
            env_manager._env_episode_count[i] == setup_async_manager_cfg['episode_num']
            for i in range(env_manager.env_num)
        )
        assert all(env_manager._env_states[i] == EnvState.DONE for i in range(env_manager.env_num))
