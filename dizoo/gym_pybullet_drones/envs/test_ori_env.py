import pytest
import gym
import numpy as np

import gym_pybullet_drones


@pytest.mark.envtest
class TestGymPybulletDronesOriEnv:

    def test_naive(self):
        env = gym.make("takeoff-aviary-v0")
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
