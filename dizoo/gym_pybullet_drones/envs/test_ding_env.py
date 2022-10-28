from easydict import EasyDict
import gym_pybullet_drones

from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env import GymPybulletDronesEnv

if __name__ == "__main__":
    cfg = {"env_id": "takeoff-aviary-v0"}
    cfg = EasyDict(cfg)
    env = GymPybulletDronesEnv(cfg)

    print(env.action_space)
    print(env.observation_space)
    print(env.reward_space)

    env.reset()

    print(env.action_space)
    print(env.observation_space)
    print(env.reward_space)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
