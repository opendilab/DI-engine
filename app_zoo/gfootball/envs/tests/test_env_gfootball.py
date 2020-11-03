import pytest
import torch
import pprint

try:
    from app_zoo.gfootball.envs.gfootball_env import GfootballEnv
except ModuleNotFoundError:
    print("[WARNING] no gfootball env, if you want to use gfootball, please install it, otherwise, ignore it.")


@pytest.mark.envtest
class TestGfootballEnv:

    def get_random_action(self, min_value, max_value):
        action = torch.randint(min_value, max_value + 1, (1, ))
        return action

    def test_naive(self):
        env = GfootballEnv({})
        print(env.info())
        reset_obs = env.reset()
        print('after reset:', reset_obs)
        pp = pprint.PrettyPrinter(indent=2)
        for i in range(3000):
            action = self.get_random_action(env.info().act_space.value['min'], env.info().act_space.value['max'])
            timestep = env.step(action)
            reward = timestep.obs
            print('reward:', reward)
            # assert reward.shape == 1
            obs = timestep.obs
            assert obs['ball_owned_team'].shape[0] == 3
            assert obs['ball_owned_player'].shape[0] == 12
            assert obs['active_player'].shape[0] == 11
            assert obs['score'].shape[0] == 22
            assert obs['steps_left'].shape[0] == 30
            print('observation: ')
            pp.pprint(obs)
            print('--step {} with action {}'.format(i, action))
        print('end')
