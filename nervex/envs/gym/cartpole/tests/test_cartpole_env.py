import pytest
import torch
import os
import random
from nervex.envs.gym.cartpole.cartpole_env import CartpoleEnv


@pytest.mark.unittest
class TestCartpoleEnv:
    def get_random_action(self, min_value, max_value):
        action = random.randint(min_value, max_value)
        return action

    def test_naive(self):
        env = CartpoleEnv({'frameskip': 2})
        print(env.info())
        obs = env.reset()
        print("obs = ", obs)
        duration = 0
        for i in range(100):
            action = self.get_random_action(env.info().act_space.value['min'], env.info().act_space.value['max'])
            timestep = env.step(action)
            duration += 1
            if timestep.done:
                env.reset()
                print("is done after {}duration".format(duration))
                duration = 0
            print('step {} with action {}'.format(i, action))
            assert (isinstance(action, int))
            print('reward {} in step {}'.format(timestep.reward, i))
            assert (isinstance(timestep.reward, float))
        print('end')
