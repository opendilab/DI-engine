import pytest
import numpy as np
from easydict import EasyDict

from dizoo.slime_volley.envs.slime_volley_env import SlimeVolleyEnv


@pytest.mark.envtest
class TestSlimeVolley:

    @pytest.mark.parametrize('agent_vs_agent', [True, False])
    def test_slime_volley(self, agent_vs_agent):
        total_rew = 0
        env = SlimeVolleyEnv(EasyDict({'env_id': 'SlimeVolley-v0', 'agent_vs_agent': agent_vs_agent}))
        # env.enable_save_replay('replay_video')
        obs1 = env.reset()
        print(env.observation_space)
        print('observation is like:', obs1)
        done = False
        while not done:
            action = env.random_action()
            observations, rewards, done, infos = env.step(action)
            if agent_vs_agent:
                total_rew += rewards[0]
            else:
                total_rew += rewards
            obs1, obs2 = observations[0], observations[1]
            assert obs1.shape == obs2.shape, (obs1.shape, obs2.shape)
            if agent_vs_agent:
                agent_lives, opponent_lives = infos[0]['ale.lives'], infos[1]['ale.lives']
        if agent_vs_agent:
            assert agent_lives == 0 or opponent_lives == 0, (agent_lives, opponent_lives)
        print("total reward is:", total_rew)
