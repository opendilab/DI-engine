import pytest
import random
import numpy as np
from dizoo.board_games.go.envs.go import raw_env  # pettingzoo go env

env = raw_env()


def policy(observation, agent):
    action = random.choice(np.flatnonzero(observation['action_mask']))
    return action


@pytest.mark.envtest
class TestGoEnv:

    def test_naive(self):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            action = policy(observation, agent) if not done else None
            env.step(action)
            # this visualizes a single game
            env.render()
