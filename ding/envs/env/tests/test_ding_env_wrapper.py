import gym
import numpy as np
import pytest
from easydict import EasyDict

from ding.torch_utils import to_ndarray
from ding.envs.env import DingEnvWrapper


class FakeEnvForTest(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(10, ), dtype=np.float32)
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(3),
                gym.spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]), shape=(2, ), dtype=np.float32)
            )
        )

    def step(self, action):
        assert self.action_space.contains(action)
        self._step_count += 1
        obs = self.observation_space.sample()
        obs = to_ndarray(obs).astype(np.float32)
        done = True if self._step_count == 100 else False
        return (obs, 0.5, done, {})

    def reset(self):
        self._step_count = 0
        obs = self.observation_space.sample()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass


gym.envs.registration.register(
    id="FakeHybridForTest-v0",
    entry_point="ding.envs.env.tests.test_ding_env_wrapper:FakeEnvForTest",
)


class TestDingEnvWrapper:

    @pytest.mark.unittest
    @pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v1'])
    def test_cartpole_pendulum(self, env_id):
        env = gym.make(env_id)
        ding_env = DingEnvWrapper(env=env)
        print(ding_env.observation_space, ding_env.action_space, ding_env.reward_space)
        cfg = EasyDict(dict(
            collector_env_num=16,
            evaluator_env_num=3,
            is_train=True,
        ))
        l1 = ding_env.create_collector_env_cfg(cfg)
        assert isinstance(l1, list)
        l1 = ding_env.create_evaluator_env_cfg(cfg)
        assert isinstance(l1, list)
        obs = ding_env.reset()
        assert isinstance(obs, np.ndarray)
        action = ding_env.random_action()
        # assert isinstance(action, np.ndarray)
        print('random_action: {}, action_space: {}'.format(action.shape, ding_env.action_space))

    @pytest.mark.envtest
    def test_mujoco(self):
        env_cfg = EasyDict(
            env_id='Ant-v3',
            env_wrapper='mujoco_default',
        )
        ding_env_mujoco = DingEnvWrapper(cfg=env_cfg)
        obs = ding_env_mujoco.reset()
        assert isinstance(obs, np.ndarray)
        # action_dim = ding_env_mujoco.action_space.shape  # n
        while True:
            # action = np.random.random(size=action_dim)  # Continuous Action
            action = ding_env_mujoco.random_action()
            timestep = ding_env_mujoco.step(action)
            # print(_, timestep.reward)
            assert timestep.reward.shape == (1, ), timestep.reward.shape
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(ding_env_mujoco.observation_space, ding_env_mujoco.action_space, ding_env_mujoco.reward_space)
        action = ding_env_mujoco.random_action()
        # assert isinstance(action, np.ndarray)
        assert action.shape == ding_env_mujoco.action_space.shape

    @pytest.mark.envtest
    @pytest.mark.parametrize('atari_env_id', ['Pong-v4', 'MontezumaRevenge-v4'])
    def test_atari(self, atari_env_id):
        env_cfg = EasyDict(
            env_id=atari_env_id,
            env_wrapper='atari_default',
        )
        ding_env_atari = DingEnvWrapper(cfg=env_cfg)

        ding_env_atari.enable_save_replay('atari_path/')
        obs = ding_env_atari.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == ding_env_atari.observation_space.shape  # (4, 84, 84)
        # action_dim = ding_env_atari.action_space.n
        while True:
            # action = np.random.choice(range(action_dim), size=(1, ))  # Discrete Action
            action = ding_env_atari.random_action()
            timestep = ding_env_atari.step(action)
            # print(timestep.reward)
            assert timestep.reward.shape == ding_env_atari.reward_space.shape, timestep.reward.shape  # (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(ding_env_atari.observation_space, ding_env_atari.action_space, ding_env_atari.reward_space)
        action = ding_env_atari.random_action()
        # assert isinstance(action, np.ndarray)
        assert action.shape == (1, )

    @pytest.mark.unittest
    @pytest.mark.parametrize('lun_bip_env_id', ['LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v3'])
    def test_lunarlander_bipedalwalker(self, lun_bip_env_id):
        env_cfg = EasyDict(
            env_id=lun_bip_env_id,
            env_wrapper='default',
        )
        ding_env_lun_bip = DingEnvWrapper(cfg=env_cfg)

        obs = ding_env_lun_bip.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == ding_env_lun_bip.observation_space.shape
        # action_space = ding_env_lun_bip.action_space
        # if lun_bip_env_id in ['LunarLanderContinuous-v2', 'BipedalWalker-v3']:
        #     action_dim = action_space.shape
        # else:
        #     action_dim = action_space.n
        while True:
            # if lun_bip_env_id in ['LunarLanderContinuous-v2', 'BipedalWalker-v3']:
            #     action = np.random.random(size=action_dim)  # Continuous Action
            # else:
            #     action = np.random.choice(range(action_dim), size=(1, ))  # Discrete Action
            action = ding_env_lun_bip.random_action()
            timestep = ding_env_lun_bip.step(action)
            # print(timestep.reward)
            assert timestep.reward.shape == ding_env_lun_bip.reward_space.shape, timestep.reward.shape  # (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(ding_env_lun_bip.observation_space, ding_env_lun_bip.action_space, ding_env_lun_bip.reward_space)
        action = ding_env_lun_bip.random_action()
        # assert isinstance(action, np.ndarray)
        print('random_action: {}, action_space: {}'.format(action.shape, ding_env_lun_bip.action_space))

    @pytest.mark.unittest
    def test_hybrid(self):
        env_cfg = EasyDict(env_id='FakeHybridForTest-v0', env_wrapper='gym_hybrid_default')
        ding_env_hybrid = DingEnvWrapper(cfg=env_cfg)

        obs = ding_env_hybrid.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == ding_env_hybrid.observation_space.shape
        while True:
            action = ding_env_hybrid.random_action()
            # print('random_action:', action)
            for k, v in action.items():
                if isinstance(v, int):
                    continue
                # print('before: {}, after: {}'.format(v.shape, ding_env_hybrid.action_space[k].shape))
                v.shape = ding_env_hybrid.action_space[k].shape
            timestep = ding_env_hybrid.step(action)
            # print(timestep.reward)
            assert timestep.reward.shape == ding_env_hybrid.reward_space.shape, timestep.reward.shape  # (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(ding_env_hybrid.observation_space, ding_env_hybrid.action_space, ding_env_hybrid.reward_space)
        action = ding_env_hybrid.random_action()
        print('random_action', action)
        assert isinstance(action, dict)
