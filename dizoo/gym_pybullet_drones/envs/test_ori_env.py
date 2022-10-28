import pytest
import gym
import numpy as np

import gym_pybullet_drones


@pytest.mark.envtest
class TestGymPybulletDronesOriEnv:
    env = gym.make("takeoff-aviary-v0")
    print(env.action_space)
    print(env.observation_space)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
