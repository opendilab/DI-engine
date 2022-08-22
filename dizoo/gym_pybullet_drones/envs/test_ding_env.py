import os
import time
import argparse
import gym
import numpy as np
from easydict import EasyDict

import gym_pybullet_drones

from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env import GymPybulletDronesEnv
# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
# from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__=="__main__":
    cfg={"env_id":"takeoff-aviary-v0"}
    cfg=EasyDict(cfg)
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
    
    done=False
    i=0
    while not done:
        i+=1
        action=env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(i)
        print(obs)
        print(reward)
        print(done)
        print(info)

