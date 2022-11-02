import pytest
from easydict import EasyDict
import gym_pybullet_drones

from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env import GymPybulletDronesEnv


@pytest.mark.envtest
class TestGymPybulletDronesEnv:
    cfg = {"env_id": "takeoff-aviary-v0"}
    cfg = EasyDict(cfg)
    env = GymPybulletDronesEnv(cfg)

    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs.shape[0] == 12
