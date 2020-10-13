import pytest
import torch

from nervex.envs.gym.pong.pong_env import PongEnv


@pytest.mark.unittest
class TestPongEnv:

    def get_random_action(self, min_value, max_value):
        action = torch.randint(min_value, max_value + 1, (1, ))
        return action

    def test_naive(self):
        env = PongEnv({'frameskip': 2, 'wrap_frame': True})
        print(env.info())
        obs = env.reset()
        print(obs)
        print("obs_shape = ", obs.shape)
        for i in range(1000):
            action = self.get_random_action(env.info().act_space.value['min'], env.info().act_space.value['max'])
            timestep = env.step(action)
            print('step {} with action {}'.format(i, action))
        print('end')
