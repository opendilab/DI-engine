from easydict import EasyDict
import pytest
import numpy as np
import pettingzoo
from ding.utils import import_module

from dizoo.petting_zoo.envs.petting_zoo_simple_tag_env import PettingZooTagEnv


@pytest.mark.envtest
class TestPettingZooEnv:

    def test_agent_obs_only(self):
        num_good=1
        num_adversaries=3
        num_obstacles=2
        env = PettingZooTagEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='simple_tag_v2',
                    num_good=num_good,
                    num_adversaries=num_adversaries,
                    num_obstacles=num_obstacles,
                    max_step=100,
                    agent_obs_only=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        # assert obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
        good_agent_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)
        adversary_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
        for key, value in obs.items():
            assert len(value) in [good_agent_obs_shape, adversary_obs_shape], f"Value of '{key}' with wrong length"
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            # assert isinstance(timestep.obs, np.ndarray), timestep.obs
            for key, value in obs.items():
                assert len(value) in [good_agent_obs_shape, adversary_obs_shape], f"Value of '{key}' with wrong length"
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
            assert timestep.reward.dtype == np.float32
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_dict_obs(self):
        num_good=1
        num_adversaries=3
        num_obstacles=2
        env = PettingZooTagEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='simple_tag_v2',
                    num_good=num_good,
                    num_adversaries=num_adversaries,
                    num_obstacles=num_obstacles,
                    max_step=100,
                    agent_obs_only=False,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        good_agent_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)
        adversary_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
        for k, v in obs.items():
            print(k, v)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, dict), timestep.obs
            assert isinstance(timestep.obs['agent_state'], dict), timestep.obs
            for key, value in timestep.obs['agent_state'].items():
                assert len(value) in [good_agent_obs_shape, adversary_obs_shape], f"Value of '{key}' with wrong length"
            assert timestep.obs['global_state'].shape == (
                (2+2)*(num_good+num_adversaries)+2*num_obstacles,
            )
            assert timestep.obs['action_mask'].dtype == np.float32
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_agent_specific_global_state(self):
        num_good=1
        num_adversaries=3
        num_obstacles=2
        good_agent_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)
        adversary_obs_shape = 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
        env = PettingZooTagEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='simple_tag_v2',
                    num_good=num_good,
                    num_adversaries=num_adversaries,
                    num_obstacles=num_obstacles,
                    max_step=100,
                    agent_obs_only=False,
                    agent_specific_global_state=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, dict), timestep.obs
            assert isinstance(timestep.obs['agent_state'], dict), timestep.obs
            for key, value in timestep.obs['agent_state'].items():
                assert len(value) in [good_agent_obs_shape, adversary_obs_shape], f"Value of '{key}' with wrong length"
            assert isinstance(timestep.obs['global_state'], dict), timestep.obs
            for key, value in timestep.obs['global_state'].items():
                assert len(value) in [good_agent_obs_shape+(2+2)*(num_good+num_adversaries)+2*num_obstacles, adversary_obs_shape+(2+2)*(num_good+num_adversaries)+2*num_obstacles], f"Value of '{key}' with wrong length"
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

# test_agent_obs_only()
# test_dict_obs()
# test_agent_specific_global_state()