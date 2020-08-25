import pytest
import torch
import os
import random
from nervex.envs.gym.pong.pong_env import PongEnv

class TestPongEnv:
    def get_random_action(self, min_value, max_value):
        action = random.randint(min_value, max_value)
        return action

    def test_naive(self):
        env = PongEnv({'frameskip':2})
        print(env.info())
        obs = env.reset()
        for i in range(10):
            action = self.get_random_action(env.info().act_space.value['min'], env.info().act_space.value['max'])
            timestep = env.step(action)
            print('step {} with action {}'.format(i, action))
        print('end')
