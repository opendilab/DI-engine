import pytest
import numpy as np
import pprint
from easydict import EasyDict

try:
    from dizoo.gfootball.envs.gfootball_academy_env import GfootballAcademyEnv
except ModuleNotFoundError:
    print("[WARNING] no gfootball env, if you want to use gfootball, please install it, otherwise, ignore it.")

cfg_keeper = EasyDict(dict(
    env_name='academy_3_vs_1_with_keeper',
    agent_num=3,
    obs_dim=26,
))

cfg_counter = EasyDict(dict(
    env_name='academy_counterattack_hard',
    agent_num=4,
    obs_dim=34,
))


@pytest.mark.envtest
class TestGfootballAcademyEnv:

    def get_random_action(self, min_value, max_value):
        action = np.random.randint(min_value, max_value + 1, (1, ))
        return action

    def test_academy_3_vs_1_with_keeper(self):
        cfg = cfg_keeper
        env = GfootballAcademyEnv(cfg)
        print(env.observation_space, env._action_space, env.reward_space)
        pp = pprint.PrettyPrinter(indent=2)
        for i in range(2):
            eps_len = 0
            # env.enable_save_replay(replay_path='./video')
            reset_obs = env.reset()
            while True:
                eps_len += 1
                action = env.random_action()[0]
                action = [int(action_agent) for k, action_agent in action.items()]
                timestep = env.step(action)
                obs = timestep.obs
                reward = timestep.reward
                done = timestep.done
                # print('observation: ')
                # pp.pprint(obs)
                assert obs['agent_state'].shape == (cfg.agent_num, cfg.obs_dim)
                assert obs['global_state'].shape == (cfg.agent_num, cfg.obs_dim * 2)
                assert obs['action_mask'].shape == (cfg.agent_num, 19)

                print('step {}, action: {}, reward: {}'.format(eps_len, action, reward))
                if done:
                    break
            assert reward == -1 or reward == 100
            print(f'Episode {i} done! The episode length is {eps_len}. The last reward is {reward}.')
        print('End')

    def test_academy_counterattack_hard(self):
        cfg = cfg_counter
        env = GfootballAcademyEnv(cfg)
        print(env.observation_space, env._action_space, env.reward_space)
        pp = pprint.PrettyPrinter(indent=2)
        for i in range(2):
            eps_len = 0
            reset_obs = env.reset()
            while True:
                eps_len += 1
                action = env.random_action()[0]
                action = [int(action_agent) for k, action_agent in action.items()]
                timestep = env.step(action)
                obs = timestep.obs
                reward = timestep.reward
                done = timestep.done
                # print('observation: ')
                # pp.pprint(obs)
                assert obs['agent_state'].shape == (cfg.agent_num, cfg.obs_dim)
                assert obs['global_state'].shape == (cfg.agent_num, cfg.obs_dim * 2)
                assert obs['action_mask'].shape == (cfg.agent_num, 19)

                print('step {}, action: {}, reward: {}'.format(eps_len, action, reward))
                if done:
                    break
            assert reward == -1 or reward == 100
            print(f'Episode {i} done! The episode length is {eps_len}. The last reward is {reward}.')
        print('End')
