import gym
import time
import pytest
import numpy as np
from ding.envs.env_manager import EnvSupervisor
from ding.envs.env_manager.env_supervisor import EnvState
from ding.framework.supervisor import ChildType


@pytest.mark.unittest
class TestEnvSupervisorCompatible:

    def test_naive(self, setup_base_manager_cfg):
        """
        To be compatible with the original env_manager, here uses the original configuration and blocking methods.
        {
            'env_cfg': [{
                'name': 'name{}'.format(i),
                'scale': 1.0,
            } for i in range(env_num)],
            'episode_num': 2,
            'reset_timeout': 10,
            'step_timeout': 8,
            'max_retry': 5,
        }
        """
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=ChildType.THREAD, env_fn=env_fn, **setup_base_manager_cfg)
        try:
            env_supervisor.seed([314 for _ in range(env_supervisor.env_num)])
            assert env_supervisor.closed
            env_supervisor.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)})

            # Test basic
            assert all([s == 314 for s in env_supervisor._env_seed.values()])

            # Test step
            count = 1
            start_time = time.time()

            # Loop over each env until done
            while not env_supervisor.done:
                env_id = env_supervisor.ready_obs_id
                action = {i: np.random.randn(4) for i in env_id}
                timestep = env_supervisor.step(action)
                assert len(timestep) == len(env_id)
                print('Count {}'.format(count))
                count += 1

            end_time = time.time()
            print('total step time: {}'.format(end_time - start_time))

            assert all([env_supervisor.env_states[env_id] == EnvState.DONE for env_id in range(env_supervisor.env_num)])
            # assert all([c == setup_base_manager_cfg.episode_num for c in env_supervisor._env_episode_count.values()])

        finally:
            # Test close
            env_supervisor.close()

            assert env_supervisor._closed
            assert all([env_supervisor.env_states[env_id] == EnvState.VOID for env_id in range(env_supervisor.env_num)])
            with pytest.raises(AssertionError):
                env_supervisor.reset([])
            with pytest.raises(AssertionError):
                env_supervisor.step([])

    def test_error(self, setup_base_manager_cfg):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=ChildType.THREAD, env_fn=env_fn, **setup_base_manager_cfg)
        try:
            # Test reset error
            with pytest.raises(RuntimeError):
                reset_param = {i: {'stat': 'error'} for i in range(env_supervisor.env_num)}
                env_supervisor.launch(reset_param=reset_param)
            assert all([state == EnvState.ERROR for state in env_supervisor.env_states.values()])
            env_supervisor.close()

            # Normal launch
            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param)
            # Normal step
            timestep = env_supervisor.step({i: np.random.randn(4) for i in range(env_supervisor.env_num)})
            assert len(timestep) == env_supervisor.env_num

            # Test reset error once, will still go correct.
            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            assert env_supervisor._retry_type == 'reset'
            reset_param[0] = {'stat': 'error_once'}
            env_supervisor.reset(reset_param)
            # assert env_supervisor.time_id[0] == env_id_0
            assert all([state == EnvState.RUN for state in env_supervisor.env_states.values()])

            return
            env_manager._retry_type = 'renew'
            env_id_0 = env_manager.time_id[0]
            reset_param[0] = {'stat': 'error_once'}
            env_manager.reset(reset_param)
            assert not env_manager._closed
            assert env_manager.time_id[0] != env_id_0

            # Test step catched error
            action = [np.random.randn(4) for i in range(env_manager.env_num)]
            action[0] = 'catched_error'
            timestep = env_manager.step(action)
            assert timestep[0].info.abnormal
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
        finally:
            env_supervisor.close()
