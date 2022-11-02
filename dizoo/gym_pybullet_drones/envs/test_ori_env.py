import pytest
import gym
import numpy as np

import gym_pybullet_drones


@pytest.mark.envtest
class TestGymPybulletDronesOriEnv:
    env = gym.make("takeoff-aviary-v0")
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs.shape[0] == 12
