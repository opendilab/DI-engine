from time import time
from numpy.core.shape_base import stack
import pytest
import numpy as np
from dizoo.overcooked.envs import OvercookEnv, OvercookGameEnv


@pytest.mark.envtest
class TestOvercooked:

    def test_overcook(self):
        concat_obs = True
        num_agent = 2
        sum_rew = 0.0
        env = OvercookEnv({'concat_obs': concat_obs})
        obs = env.reset()
        for _ in range(env._horizon):
            action = np.random.randint(0, 6, (num_agent, ))
            timestep = env.step(action)
            obs = timestep.obs
            for k, v in obs.items():
                if k not in ['agent_state', 'action_mask']:
                    continue
                if concat_obs:
                    assert v.shape == env.info().obs_space.shape[k]
                else:
                    assert len(v) == len(env.info().obs_space.shape[k])
        assert timestep.done
        sum_rew += timestep.info['final_eval_reward'][0]
        print("sum reward is:", sum_rew)

    def test_overcook_game(self):
        concat_obs = False
        num_agent = 2
        env = OvercookGameEnv({'concat_obs': concat_obs})
        obs = env.reset()
        for _ in range(env._horizon):
            action = [np.random.randint(0, 6), np.random.randint(0, 6)]
            timestep = env.step(action)
            obs = timestep.obs
            print("shaped reward is:", timestep.info[0]['shaped_r_by_agent'], timestep.info[1]['shaped_r_by_agent'])
        assert timestep.done
        print("agent 0 sum reward is:", timestep.info[0]['final_eval_reward'])
        print("agent 1 sum reward is:", timestep.info[1]['final_eval_reward'])
