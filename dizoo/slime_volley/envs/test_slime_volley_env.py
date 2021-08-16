import pytest
import numpy as np
from easydict import EasyDict

from dizoo.slime_volley.envs.slime_volley_env import SlimeVolleyEnv


@pytest.mark.unittest
class TestSlimeVolley:

    def test_slime_volley(self):
        total_rew = 0
        env = SlimeVolleyEnv(EasyDict({'env_id': 'SlimeVolley-v0'}))
        obs1 = env.reset()
        done = False
        print(env._env.observation_space)
        print('observation is like:', obs1)
        done = False
        while not done:
            action1 = np.random.randint(0, 2, (3, ))
            action2 = np.random.randint(0, 2, (3, ))
            print(action1.sum(), action2.sum())
            observations, rewards, done, infos = env.step([action1, action2])
            total_rew += rewards[0]
            obs1, obs2 = observations[0], observations[1]
            assert obs1.shape == obs2.shape, (obs1.shape, obs2.shape)
            agent_lives, opponent_lives = infos[0]['ale.lives'], infos[0]['ale.otherLives']
        assert agent_lives == 0 or opponent_lives == 0, (agent_lives, opponent_lives)
        print("total reward is:", total_rew)

        # == info in slime volley env ==
        # info = {
        # 'ale.lives': agent's lives left,
        # 'ale.otherLives': opponent's lives left,
        # 'otherObs': opponent's observations,
        # 'state': agent's state (same as obs in state mode),
        # 'otherState': opponent's state (same as otherObs in state mode),
        # }
