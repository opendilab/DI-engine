import pytest
import numpy as np
from easydict import EasyDict

from dizoo.slime_volley.envs.slime_volley_env import SlimeVolleyEnv


@pytest.mark.unittest
class TestOvercooked:

    def test_slam_volley(self):
        sum_rew = 0
        num_agent = 2
        env = SlimeVolleyEnv(EasyDict({'env_id': 'SlimeVolley-v0'}))
        obs = env.reset()
        done = False
        print(env._env.observation_space)
        exit(0)
        for _ in range(env._horizon):
            action = np.random.randint(0, 6, (num_agent, ))
            timestep = env.step(action)
            obs = timestep.obs
            for k, v in obs.items():
                if k not in ['agent_state', 'action_mask']:
                    continue
                else:
                    assert len(v) == len(env.info().obs_space.shape[k])
        assert timestep.done
        sum_rew += timestep.info['final_eval_reward'][0]
        print("sum reward is:", sum_rew)



