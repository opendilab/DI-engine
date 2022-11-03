import pytest
from easydict import EasyDict
import gym_pybullet_drones

from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env import GymPybulletDronesEnv


@pytest.mark.envtest
class TestGymPybulletDronesEnv:

    def test_naive(self):
        cfg = {"env_id": "takeoff-aviary-v0"}
        cfg = EasyDict(cfg)
        env = GymPybulletDronesEnv(cfg)

        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            assert action.shape[0] == 4

            for i in range(action.shape[0]):
                assert action[i] >= env.action_space.low[i] and action[i] <= env.action_space.high[i]

            obs, reward, done, info = env.step(action)

            assert obs.shape[0] == 12
            for i in range(obs.shape[0]):
                assert obs[i] >= env.observation_space.low[i] and obs[i] <= env.observation_space.high[i]

            assert reward >= env.reward_space.low and reward <= env.reward_space.high
