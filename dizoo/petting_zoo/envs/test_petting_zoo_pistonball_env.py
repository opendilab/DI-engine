from easydict import EasyDict
import pytest
import numpy as np
from dizoo.petting_zoo.envs.petting_zoo_pistonball_env import PettingZooPistonballEnv


@pytest.mark.envtest
class TestPettingZooPistonballEnv:

    def test_agent_obs_only(self):
        n_pistons = 20
        env = PettingZooPistonballEnv(
            EasyDict(
                dict(
                    n_pistons=n_pistons,
                    max_cycles=125,
                    agent_obs_only=True,
                    continuous_actions=True,
                    act_scale=False,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        assert obs.shape == (n_pistons, 3, 457, 120)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            # print(timestep)
            assert isinstance(timestep.obs, np.ndarray), timestep.obs
            assert timestep.obs.shape == (n_pistons, 3, 457, 120)
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
            assert timestep.reward.dtype == np.float32
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_dict_obs(self):
        n_pistons = 20
        env = PettingZooPistonballEnv(
            EasyDict(
                dict(
                    n_pistons=n_pistons,
                    max_cycles=125,
                    agent_obs_only=False,
                    agent_specific_global_state=False,
                    continuous_actions=True,
                    act_scale=False,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v.shape)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            # print(timestep)
            assert isinstance(timestep.obs['agent_state'], np.ndarray), timestep.obs['agent_state']
            assert isinstance(timestep.obs['global_state'], np.ndarray), timestep.obs['global_state']
            assert timestep.obs['agent_state'].shape == (n_pistons, 3, 457, 120)
            assert timestep.obs['global_state'].shape == (3, 560, 880)
            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()

    def test_agent_specific_global_state(self):
        n_pistons = 20
        env = PettingZooPistonballEnv(
            EasyDict(
                dict(
                    n_pistons=n_pistons,
                    max_cycles=125,
                    agent_obs_only=False,
                    continuous_actions=True,
                    agent_specific_global_state=True,
                    act_scale=False,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v.shape)
        for i in range(10):
            random_action = env.random_action()
            random_action = np.array([random_action[agent] for agent in random_action])
            timestep = env.step(random_action)
            # print(timestep)
            assert isinstance(timestep.obs['agent_state'], np.ndarray), timestep.obs['agent_state']
            assert isinstance(timestep.obs['global_state'], np.ndarray), timestep.obs['global_state']
            assert timestep.obs['agent_state'].shape == (n_pistons, 3, 457, 120)
            assert timestep.obs['global_state'].shape == (n_pistons, 3, 560, 880)
            assert timestep.obs['global_state'].shape == (n_pistons, 3, 560, 880)

            assert isinstance(timestep.done, bool), timestep.done
            assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()