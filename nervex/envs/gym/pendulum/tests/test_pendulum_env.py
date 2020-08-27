import pytest
import torch
import os
import random
from nervex.envs.gym.pendulum.pendulum_env import PendulumEnv


@pytest.mark.unittest
class TestPendulumEnv:
    def get_random_action(self, min_value, max_value):
        action = random.uniform(min_value, max_value)
        return action

    def test_naive(self):
        env = PendulumEnv({'frameskip': 2})
        print(env.info())
        obs = env.reset()
        print("obs = ", obs)
        for i in range(100):
            action = self.get_random_action(env.info().act_space.value['min'], env.info().act_space.value['max'])
            timestep = env.step(action)
            print('step {} with action {}'.format(i, action))
            print('reward {} in step {}'.format(timestep.reward, i))
        print('end')
