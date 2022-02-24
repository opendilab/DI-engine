import numpy as np
import pettingzoo
from ding.utils import import_module
from easydict import EasyDict
import pytest

from dizoo.petting_zoo.envs.petting_zoo_env import PettingZooEnv

# 1
# from pettingzoo.mpe import simple_v2
# env1 = simple_v2.env()
# print(env1)
# print(type(env1))

# 2
# import_module(['pettingzoo.mpe.simple_spread_v2'])
# env2 = pettingzoo.__dict__['mpe'].__dict__['simple_spread_v2'].parallel_env(N=4)
# adict = pettingzoo.__dict__['mpe'].__dict__
# print(env2)
# print(type(env2))
# env2.reset()
# # print('agents', env2.agents, env2.num_agents, env2.agent_selection)
# # for agent in env2.agents:
# #     print(agent, 'obs_space:{}, act_space:{}'.format(env2.observation_space(agent), env2.action_space(agent)))
# # print('dones', env2.dones)
# # print('infos', env2.infos)
# # print('rewards', env2.rewards)
# # print('obs_space type', type(env2.observation_space('agent_0')))
# # print('act_space type', type(env2.action_space('agent_0')))
# for _ in range(5):
#     # print(env2.agent_selection, env2.last()[0].shape)
#     obs, rew, dones, infos = env2.step({'agent_' + str(i): 0 for i in range(4)})
#     print(obs)


@pytest.mark.envtest
class TestPettingZooEnv:

    # def test_agent_obs_only(self):
    #     n_agent = 5
    #     n_landmark = n_agent
    #     env = PettingZooEnv(
    #         EasyDict(
    #             dict(
    #                 env_family='mpe',
    #                 env_id='simple_spread_v2',
    #                 n_agent=n_agent,
    #                 n_landmark=n_landmark,
    #                 max_step=100,
    #                 agent_obs_only=True,
    #                 continuous_actions=True,
    #             )
    #         )
    #     )
    #     env.seed(123)
    #     assert env._seed == 123
    #     obs = env.reset()
    #     assert obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
    #     for _ in range(10):
    #         random_action = env.action_space.sample()
    #         random_action = np.array([random_action[agent] for agent in random_action])
    #         timestep = env.step(random_action)
    #         print(timestep)
    #         assert isinstance(timestep.obs, np.ndarray), timestep.obs
    #         assert timestep.obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
    #         assert isinstance(timestep.done, bool), timestep.done
    #         assert isinstance(timestep.reward, np.ndarray), timestep.reward
    #     print(env.observation_space, env.action_space, env.reward_space)
    #     env.close()

    def test_dict_obs(self):
        n_agent = 5
        n_landmark = n_agent
        env = PettingZooEnv(
            EasyDict(
                dict(
                    env_family='mpe',
                    env_id='simple_spread_v2',
                    n_agent=n_agent,
                    n_landmark=n_landmark,
                    max_step=100,
                    agent_obs_only=False,
                    continuous_actions=True,
                )
            )
        )
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for k, v in obs.items():
            print(k, v.shape)
        # assert obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
        # for _ in range(10):
        #     random_action = env.action_space.sample()
        #     random_action = np.array([random_action[agent] for agent in random_action])
        #     timestep = env.step(random_action)
        #     print(timestep)
        #     assert isinstance(timestep.obs, np.ndarray), timestep.obs
        #     assert timestep.obs.shape == (n_agent, 2 + 2 + (n_agent - 1) * 2 + n_agent * 2 + (n_agent - 1) * 2)
        #     assert isinstance(timestep.done, bool), timestep.done
        #     assert isinstance(timestep.reward, np.ndarray), timestep.reward
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
