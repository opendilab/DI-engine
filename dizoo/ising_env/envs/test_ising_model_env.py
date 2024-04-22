import pytest
import numpy as np
from dizoo.ising_env.envs import IsingModelEnv
from easydict import EasyDict

num_agents = 100


@pytest.mark.envtest
class TestIsingModelEnv:

    def test_ising(self):
        env = IsingModelEnv(EasyDict({'num_agents': num_agents, 'dim_spin': 2, 'agent_view_sight': 1}))
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (num_agents, 4 + 2)
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
            for i in range(10):
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.random.randint(0, env.action_space.n, size=(num_agents, 1))
                else:
                    random_action = np.array([env.action_space.sample() for _ in range(num_agents)])
                    random_action = np.expand_dims(random_action, axis=1)
                timestep = env.step(random_action)
                print('timestep', timestep, '\n')
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (num_agents, 4 + 2)
                assert timestep.reward.shape == (num_agents, 1)
                assert timestep.reward[0] >= env.reward_space.low
                assert timestep.reward[0] <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
