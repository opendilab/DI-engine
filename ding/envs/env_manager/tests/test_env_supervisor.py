import time
import pytest
import numpy as np
import treetensor.numpy as tnp
from ding.envs.env_manager import EnvSupervisor
from ding.envs.env_manager.env_supervisor import EnvState
from ding.framework.supervisor import ChildType
from gym.spaces import Space


class TestEnvSupervisorCompatible:
    "Test compatibility with base env manager."

    @pytest.mark.unittest
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
            print('Total step time: {}'.format(end_time - start_time))

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

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_reset_error(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        # Test reset error
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'error'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param)
        assert env_supervisor.closed

    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_reset_error_once(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
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
        env_supervisor.close()

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_renew_error(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **{**setup_base_manager_cfg, "retry_type": "renew"})
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
        for i, obs in enumerate(env_supervisor.ready_obs):
            assert all(x == y for x, y in zip(obs, env_supervisor._ready_obs.get(i)))

        # Test step catched error
        action = [np.random.randn(4) for i in range(env_supervisor.env_num)]
        action[0] = 'catched_error'
        timestep = env_supervisor.step(action)
        assert timestep[0].info.abnormal

        assert all(['abnormal' not in timestep[i].info for i in range(1, env_supervisor.env_num)])
        # With auto_reset, abnormal timestep with done==True will be auto reset.
        assert all([env_supervisor.env_states[i] == EnvState.RUN for i in range(env_supervisor.env_num)])
        assert len(env_supervisor.ready_obs) == 4
        env_supervisor.close()

    @pytest.mark.tmp  # gitlab ci and local test pass, github always fail
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_block_launch(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg['max_retry'] = 1
        setup_base_manager_cfg['reset_timeout'] = 7

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

    @pytest.mark.tmp  # gitlab ci and local test pass, github always fail
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_block_step(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg['max_retry'] = 1
        setup_base_manager_cfg['reset_timeout'] = 7

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

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_properties(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        env_supervisor.launch(reset_param=reset_param)

        assert isinstance(env_supervisor.action_space, Space)
        assert isinstance(env_supervisor.reward_space, Space)
        assert isinstance(env_supervisor.observation_space, Space)
        env_supervisor.close()

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_auto_reset(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(
            type_=type_, env_fn=env_fn, **{
                **setup_base_manager_cfg, "auto_reset": True,
                "episode_num": 1000
            }
        )
        env_supervisor.launch(reset_param={i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)})

        assert len(env_supervisor.ready_obs) == 4
        assert len(env_supervisor.ready_obs_id) == 4

        timesteps = []

        for _ in range(10):
            action = {i: np.random.randn(4) for i in range(env_supervisor.env_num)}
            timesteps.append(env_supervisor.step(action))
            assert len(env_supervisor.ready_obs) == 4
            time.sleep(1)
        timesteps = tnp.stack(timesteps).reshape(-1)
        assert len(timesteps.done) == 40
        assert any(done for done in timesteps.done)
        assert all([env_supervisor.env_states[env_id] == EnvState.RUN for env_id in range(env_supervisor.env_num)])
        env_supervisor.close()


class TestEnvSupervisor:
    """
    Test async usage
    """

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_normal(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg["auto_reset"] = False
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        env_supervisor.seed([314 for _ in range(env_supervisor.env_num)])
        env_supervisor.launch(
            reset_param={i: {
                'stat': 'stat_test'
            }
                         for i in range(env_supervisor.env_num)}, block=False
        )

        count = 0
        start_time = time.time()
        while not env_supervisor.done:
            recv_payload = env_supervisor.recv()
            if recv_payload.method == "reset":  # Recv reset obs
                assert len(recv_payload.data) == 3
            elif recv_payload.method == "step":
                assert isinstance(recv_payload.data, tnp.ndarray)
            if env_supervisor.env_states[recv_payload.proc_id] != EnvState.DONE:
                action = {recv_payload.proc_id: np.random.randn(4)}
                env_supervisor.step(action, block=False)
            count += 1
            print("Count", count)

        end_time = time.time()
        print("Total step time: {}".format(end_time - start_time))

        env_supervisor.close()
        assert env_supervisor.closed

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_reset_error(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'error'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param, block=False)
            while True:
                env_supervisor.recv()
        env_supervisor.close()

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_reset_error_once(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        # Normal launch
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        env_supervisor.launch(reset_param=reset_param)

        env_id_0 = env_supervisor.time_id[0]

        # Normal step
        env_supervisor.step({i: np.random.randn(4) for i in range(env_supervisor.env_num)}, block=False)
        timestep = []
        while len(timestep) != 4:
            payload = env_supervisor.recv()
            if payload.method == "step":
                timestep.append(payload.data)
        assert len(timestep) == env_supervisor.env_num

        # Test reset error once, will still go correct.
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        assert env_supervisor._retry_type == 'reset'
        reset_param[0] = {'stat': 'error_once'}
        env_supervisor.reset(reset_param, block=False)  # First try, success
        env_supervisor.reset(reset_param, block=False)  # Second try, error and recover

        reset_obs = []
        while len(reset_obs) != 8:
            reset_obs.append(env_supervisor.recv(ignore_err=True))
        assert env_supervisor.time_id[0] == env_id_0
        assert all([state == EnvState.RUN for state in env_supervisor.env_states.values()])
        env_supervisor.close()

    @pytest.mark.unittest
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_renew_error_once(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg["retry_type"] = "renew"
        setup_base_manager_cfg["shared_memory"] = False
        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        # Normal launch
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        env_supervisor.launch(reset_param=reset_param)

        env_id_0 = env_supervisor.time_id[0]
        reset_param[0] = {'stat': 'error_once'}
        env_supervisor.reset(reset_param, block=False)
        env_supervisor.reset(reset_param, block=False)

        reset_obs = []
        while len(reset_obs) != 8:
            reset_obs.append(env_supervisor.recv(ignore_err=True))

        assert env_supervisor.time_id[0] != env_id_0
        assert len(env_supervisor.ready_obs) == 4

        # Test step catched error
        action = [np.random.randn(4) for i in range(env_supervisor.env_num)]
        action[0] = 'catched_error'
        env_supervisor.step(action, block=False)

        timestep = {}
        while len(timestep) != 4:
            payload = env_supervisor.recv()
            if payload.method == "step":
                timestep[payload.proc_id] = payload.data
        assert len(timestep) == env_supervisor.env_num
        assert timestep[0].info.abnormal

        assert all(['abnormal' not in timestep[i].info for i in range(1, env_supervisor.env_num)])
        env_supervisor.close()

    @pytest.mark.tmp  # gitlab ci and local test pass, github always fail
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_block_launch(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg["retry_type"] = "renew"
        setup_base_manager_cfg['max_retry'] = 1
        setup_base_manager_cfg['reset_timeout'] = 7

        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        with pytest.raises(RuntimeError):
            reset_param = {i: {'stat': 'block'} for i in range(env_supervisor.env_num)}
            env_supervisor.launch(reset_param=reset_param, block=False)
            while True:
                payload = env_supervisor.recv()
        assert env_supervisor.closed

        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        reset_param[0]['stat'] = 'wait'

        env_supervisor.launch(reset_param=reset_param, block=False)

        reset_obs = []
        while len(reset_obs) != 4:
            payload = env_supervisor.recv(ignore_err=True)
            if payload.method == "reset":
                reset_obs.append(payload.data)

        env_supervisor.close(1)

    @pytest.mark.tmp  # gitlab ci and local test pass, github always fail
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
    def test_block_step(self, setup_base_manager_cfg, type_):
        env_fn = setup_base_manager_cfg.pop('env_fn')
        setup_base_manager_cfg["retry_type"] = "renew"
        setup_base_manager_cfg['max_retry'] = 1
        setup_base_manager_cfg['reset_timeout'] = 7

        env_supervisor = EnvSupervisor(type_=type_, env_fn=env_fn, **setup_base_manager_cfg)
        reset_param = {i: {'stat': 'stat_test'} for i in range(env_supervisor.env_num)}
        env_supervisor.launch(reset_param=reset_param)

        # Block step will reset env, thus cause runtime error
        env_supervisor._reset_param[0] = {"stat": "block"}
        # Test step timeout
        action = [np.random.randn(4) for i in range(env_supervisor.env_num)]
        action[0] = 'block'

        with pytest.raises(RuntimeError):
            env_supervisor.step(action, block=False)
            while True:
                env_supervisor.recv()
        assert env_supervisor.closed

        env_supervisor.launch(reset_param)
        action[0] = 'wait'
        env_supervisor.step(action, block=False)
        timestep = []
        while len(timestep) != 4:
            payload = env_supervisor.recv(ignore_err=True)
            if payload.method == "step":
                timestep.append(payload.data)

        env_supervisor.close(1)
