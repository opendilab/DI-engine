import time
import pytest
import numpy as np
from ding.envs.env_manager import EnvSupervisor
from ding.envs.env_manager.env_supervisor import EnvState
from ding.framework.supervisor import ChildType


@pytest.mark.unittest
class TestEnvSupervisorCompatible:

    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_naive(self, setup_base_manager_cfg, type_):
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
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **{**setup_base_manager_cfg, "auto_reset": False})
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

        finally:
            # Test close
            env_supervisor.close()

            assert env_supervisor.closed
            assert all([env_supervisor.env_states[env_id] == EnvState.VOID for env_id in range(env_supervisor.env_num)])
            with pytest.raises(AssertionError):
                env_supervisor.reset([])
            with pytest.raises(AssertionError):
                env_supervisor.step([])

    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_error(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')

        def test_reset_error():
            env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
            # Test reset error
            with pytest.raises(RuntimeError):
                reset_param = {i: {'stat': 'error'} for i in range(env_supervisor.env_num)}
                env_supervisor.launch(reset_param=reset_param)
            assert env_supervisor.closed

        def test_reset_error_once():
            env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
            # Normal launch
            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param)

            env_id_0 = env_supervisor.time_id[0]
            # Normal step
            timestep = env_supervisor.step({i: np.random.randn(4) for i in range(env_supervisor.env_num)})
            assert len(timestep) == env_supervisor.env_num

            # Test reset error once, will still go correct.
            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            assert env_supervisor._retry_type == 'reset'
            reset_param[0] = {'stat': 'error_once'}
            env_supervisor.reset(reset_param)
            env_supervisor.reset(reset_param)

            # If retry type is reset, time id should be equal
            assert env_supervisor.time_id[0] == env_id_0
            assert all([state == EnvState.RUN for state in env_supervisor.env_states.values()])

        def test_renew_error():
            env_supervisor = EnvSupervisor(
                type_=type_, env_fn=env_fn, **{
                    **setup_base_manager_cfg, "retry_type": "renew"
                }
            )
            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param)

            assert env_supervisor._retry_type == "renew"
            env_id_0 = env_supervisor.time_id[0]

            reset_param[0] = {'stat': 'error_once'}
            env_supervisor.reset(reset_param)
            env_supervisor.reset(reset_param)
            assert not env_supervisor.closed
            # If retry type is renew, time id should not be equal
            assert env_supervisor.time_id[0] != env_id_0
            assert len(env_supervisor.ready_obs) == 4

            # Test step catched error
            action = [np.random.randn(4) for i in range(env_supervisor.env_num)]
            action[0] = 'catched_error'
            timestep = env_supervisor.step(action)
            assert timestep[0].info.abnormal

            assert all(['abnormal' not in timestep[i].info for i in range(1, env_supervisor.env_num)])
            # With auto_reset, abnormal timestep with done==True will be auto reset.
            assert all([env_supervisor.env_states[i] == EnvState.RUN for i in range(env_supervisor.env_num)])
            assert len(env_supervisor.ready_obs) == 4

        test_reset_error()
        test_reset_error_once()
        test_renew_error()

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_block(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg['max_retry'] = 1
        setup_base_manager_cfg['reset_timeout'] = 7

        def test_block_launch():
            env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
            with pytest.raises(RuntimeError):
                reset_param = {i: {'stat': 'block'} for i in range(env_supervisor.env_num)}
                env_supervisor.launch(reset_param=reset_param)
            assert env_supervisor.closed

            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            reset_param[0]['stat'] = 'wait'

            env_supervisor.launch(reset_param=reset_param)
            assert not env_supervisor.closed

            env_supervisor.close(1)

        def test_block_step():
            env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)

            reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param)

            timestep = env_supervisor.step({i: np.random.randn(4) for i in range(env_supervisor.env_num)})
            assert len(timestep) == env_supervisor.env_num

            # Block step will reset env, thus cause runtime error
            env_supervisor._reset_param[0] = {"stat": "block"}
            # Test step timeout
            action = [np.random.randn(4) for i in range(env_supervisor.env_num)]
            action[0] = 'block'

            with pytest.raises(RuntimeError):
                timestep = env_supervisor.step(action)
            assert env_supervisor.closed

            env_supervisor.launch(reset_param)
            action[0] = 'wait'
            timestep = env_supervisor.step(action)
            assert len(timestep) == env_supervisor.env_num

            env_supervisor.close(1)

        test_block_launch()
        test_block_step()


@pytest.mark.unittest
class TestEnvSupervisor:
    """
    Test async apis
    """

    def test_naive(self, setup_base_manager_cfg):
        pass
