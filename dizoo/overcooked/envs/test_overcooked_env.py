from time import time
from easydict import EasyDict
import pytest
import numpy as np
from dizoo.overcooked.envs import OvercookEnv, OvercookGameEnv


@pytest.mark.envtest
class TestOvercooked:

    @pytest.mark.parametrize("action_mask", [True, False])
    def test_overcook(self, action_mask):
        num_agent = 2
        sum_rew = 0.0
        env = OvercookEnv(EasyDict({'concat_obs': True, 'action_mask': action_mask}))
        obs = env.reset()
        for _ in range(env._horizon):
            action = env.random_action()
            timestep = env.step(action)
            obs = timestep.obs
            if action_mask:
                for k, v in obs.items():
                    if k not in ['agent_state', 'action_mask']:
                        assert False
                    assert v.shape == env.observation_space[k].shape
            else:
                assert obs.shape == env.observation_space.shape
        assert timestep.done
        sum_rew += timestep.info['final_eval_reward'][0]
        print("sum reward is:", sum_rew)

    @pytest.mark.parametrize("concat_obs", [True, False])
    def test_overcook_game(self, concat_obs):
        env = OvercookGameEnv(EasyDict({'concat_obs': concat_obs}))
        print('observation space: {}'.format(env.observation_space.shape))
        obs = env.reset()
        for _ in range(env._horizon):
            action = env.random_action()
            timestep = env.step(action)
            obs = timestep.obs
            assert obs.shape == env.observation_space.shape
        assert timestep.done
        print("agent 0 sum reward is:", timestep.info[0]['final_eval_reward'])
        print("agent 1 sum reward is:", timestep.info[1]['final_eval_reward'])
