import pytest
import torch

from app_zoo.cartpole.envs import CartpoleEnv


@pytest.mark.envtest
class TestCartpoleEnv:

    def get_random_action(self, min_value, max_value):
        action = torch.randint(min_value, max_value, (1, ))
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
            # assert (isinstance(action.dtype, torch.int64))
            assert isinstance(action, torch.LongTensor)
            print('reward {} in step {}'.format(timestep.reward, i))
            assert (isinstance(timestep.reward, torch.FloatTensor))
        print('end')
