"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/test.py
"""
import pytest
from easydict import EasyDict
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast

import ding.rl_utils.mcts.ctree.cytree as cytree
from ding.rl_utils.mcts.game import GameHistory
from ding.rl_utils.mcts.utils import select_action, prepare_observation_lst
# from ding.rl_utils.mcts.mcts_ptree import MCTS
from ding.rl_utils.mcts.mcts_ctree import MCTS

args = ['PongNoFrameskip-v4', 'tictactoe']


@pytest.mark.unittest
@pytest.mark.parametrize('env_name', args)
def test_game_history(env_name):
    if env_name == 'PongNoFrameskip-v4':
        from dizoo.board_games.atari.config.atari_config import game_config
    elif env_name == 'tictactoe':
        from dizoo.board_games.tictactoe.config.tictactoe_config import game_config
    config = game_config

    # set some additional config for test
    config.device = 'cpu'
    config.evaluator_env_num = 2
    config.render = False
    config.max_episode_steps = int(1e2)
    config.num_simulations = 2
    config.game_history_max_length = 20

    # to obtain model
    # model = EfficientZeroNet()
    model = config.get_uniform_network()
    model.to(config.device)
    model.eval()

    with torch.no_grad():
        if env_name == 'PongNoFrameskip-v4':
            from dizoo.atari.envs.atari_muzero_env import AtariMuZeroEnv
            envs = [AtariMuZeroEnv(config) for i in range(config.evaluator_env_num)]
        elif env_name == 'tictactoe':
            from dizoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
            envs = [TicTacToeEnv(config) for i in range(config.evaluator_env_num)]

        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(config.evaluator_env_num)])
        game_histories = [
            GameHistory(envs[i].action_space, max_length=config.game_history_max_length, config=config)
            for i in range(config.evaluator_env_num)
        ]
        for i in range(config.evaluator_env_num):
            game_histories[i].init([init_obses[i]['observation'] for _ in range(config.stacked_observations)])

        ep_ori_rewards = np.zeros(config.evaluator_env_num)
        ep_clip_rewards = np.zeros(config.evaluator_env_num)
        # loop
        while not dones.all():
            if config.render:
                for i in range(config.evaluator_env_num):
                    envs[i].render()
            stack_obs = [game_history.step_obs() for game_history in game_histories]
            stack_obs = prepare_observation_lst(stack_obs)
            if config.image_based:
                stack_obs = torch.from_numpy(stack_obs).to(config.device).float() / 255.0
            else:
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(config.device)
            with autocast():
                # stack_obs {Tensor:(2,12,96,96)}
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state  # {ndarray:（2, 64, 6, 6）}
            reward_hidden_roots = network_output.reward_hidden  # {tuple:2}->{ndarray:(1,2,512)}
            value_prefix_pool = network_output.value_prefix  # {list: 2}->{float}
            policy_logits_pool = network_output.policy_logits.tolist()  # {list: 2}->{list:6}->{float}

            roots = cytree.Roots(config.evaluator_env_num, config.action_space_size, config.num_simulations)
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)
            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_values = roots.get_values()  # {list: 1}
            for i in range(config.evaluator_env_num):
                if dones[i]:
                    continue
                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                obs, ori_reward, done, info = env.step(action)
                obs = obs['observation']
                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

        for env in envs:
            env.close()
