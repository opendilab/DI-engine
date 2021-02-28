import pytest
import torch
import os
import random
from random import choice
from app_zoo.boardgame.gobang.gobang_env import GobangEnv

import numpy as np


@pytest.mark.unittest
class TestGobangEnv:

    def get_random_action(self, min_value, max_value):
        action = random.randint(min_value, max_value)
        return action

    def select_from_actions(self, env):
        action = choice(env.get_available())
        return action

    def test_naive(self):
        env = GobangEnv({'place_holder': None})
        print(env.info())
        obs = env.reset()
        print("obs = ", obs)
        duration = 0
        for i in range(1000):
            duration += 1
            action = self.select_from_actions(env)
            action = np.array([action])
            timestep = env.step(action)
            print('step {} with action {}'.format(i, action))
            print('reward {} in step {}'.format(timestep.reward, i))
            if timestep.done:
                print("Done now: ", timestep)
                env.reset()
                print("is done after {} duration".format(duration))
                duration = 0
        print('end')

    def test_mask(self):
        env = GobangEnv({'place_holder': None})
        print(env.info())
        obs = env.reset()
        print("obs = ", obs)
        duration = 0
        for i in range(1000):
            duration += 1
            action = self.get_random_action(0, 63)
            while obs['action_mask'][action] == 0:
                action = self.get_random_action(0, 63)
            action = np.array([action])
            timestep = env.step(action)
            obs = timestep.obs
            print('step {} with action {}'.format(i, action))
            print('reward {} in step {}'.format(timestep.reward, i))
            if timestep.done:
                print("Done now: ", timestep)
                obs = env.reset()
                print("is done after {} duration".format(duration))
                duration = 0
        print('end')