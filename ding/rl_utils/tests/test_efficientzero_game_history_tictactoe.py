"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/test.py
"""
import pytest
from easydict import EasyDict
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list
import ding.rl_utils.efficientzero.ctree.cytree as cytree
from ding.rl_utils.efficientzero.game import GameHistory
from ding.rl_utils.efficientzero.utils import select_action, prepare_observation_lst
from ding.rl_utils.efficientzero.mcts import MCTS
from dizoo.board_games.tictactoe.envs import TicTacToeEnv
from dizoo.board_games.tictactoe.config.tictactoe_config import game_config as config
from ding.model.template.model_based.efficientzero_tictactoe_model import EfficientZeroNet
# from ding.model.template.model_based.efficientzero_atari_model import EfficientZeroNet   # TODO

GameBuffer_config = EasyDict(
    dict(
        batch_size=10,
        transition_num=20,
        priority_prob_alpha=0.5,
        total_transitions=10000,
    )
)


@pytest.mark.unittest
def test_game_history():
    args = EasyDict(
        dict(
            env='PongNoFrameskip-v4',
            seed=0,
            render=False,
            use_priority=False,
            debug=False,
            case='atari',
            opr='train',
            amp_type='none',
            # amp_type='torch_amp',
            num_gpus=0,
            num_cpus=16,
            cpu_actor=2,
            gpu_actor=0,
            object_store_memory=1 * 1024 * 1024 * 1024,
            no_cuda=True,
            device='cpu',
            p_mcts_num=8,
            use_root_value=False,
            use_augmentation=True,
            augmentation=['shift', 'intensity'],
            # result_dir=os.path.join(os.getcwd(), 'results'),
            result_dir='/Users/puyuan/code/DI-engine/results',
            revisit_policy_search_rate=0.99,
            info='',
            load_model=False,
            model_path='./results/test_model.p',
            num_simulations=2,  # TODO
            # UCB formula
            pb_c_base=19652,
            pb_c_init=1.25,
            discount=0.997,
        )
    )

    # set config as per arguments
    exp_path = config.set_config(args)  # TODO

    done = False
    render = False
    save_video = False
    final_test = False
    use_pb = True
    counter = 0
    device = args.device
    test_episodes = 2
    config.max_moves = 20
    config.stacked_observations = 4
    config.cvt_string = False
    model = EfficientZeroNet(
        observation_shape=(3, 3, 3),
        action_space_size=9,
        num_blocks=1,
        # num_channels=64,
        num_channels=12,
        reduced_channels_reward=16,
        reduced_channels_value=16,
        reduced_channels_policy=16,
        fc_reward_layers=[32],
        fc_value_layers=[32],
        fc_policy_layers=[32],
        reward_support_size=32,
        value_support_size=32,
        downsample=False,
    )
    # model = config.get_uniform_network()

    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))

    if use_pb:
        pb = tqdm(np.arange(config.max_moves), leave=True)

    with torch.no_grad():
        # new games
        # envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
        #                         video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]

        envs = [TicTacToeEnv() for i in range(test_episodes)]

        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(envs[_].action_space, max_length=config.max_moves, config=config) for _ in range(test_episodes)
        ]
        # for i in range(test_episodes):
        #     game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])
        step = 0
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)
        # loop
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            if config.image_based:
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                stack_obs = to_ndarray(stack_obs)  # TODO

                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            with autocast():
                # stack_obs {Tensor:(2,12,96,96)}
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state  # {ndarray:（2, 64, 6, 6）}
            reward_hidden_roots = network_output.reward_hidden  # {tuple:2}->{ndarray:(1,2,512)}
            value_prefix_pool = network_output.value_prefix  # {list: 2}->{float}
            policy_logits_pool = network_output.policy_logits.tolist()  # {list: 2}->{list:6}->{float}

            roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_values = roots.get_values()  # {list: 1}
            for i in range(test_episodes):
                if dones[i]:
                    continue

                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)

                obs, ori_reward, done, info = env.step(action)
                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

            step += 1
            if use_pb:
                pb.set_description(
                    '{} In step {}, scores: {}(max: {}, min: {}) currently.'
                    ''.format(
                        config.env_name, counter, ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()
                    )
                )
                pb.update(1)

        for env in envs:
            env.close()
