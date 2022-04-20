"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/test.py
"""

from easydict import EasyDict
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast

from ding.torch_utils import to_list
import ding.rl_utils.efficientzero.ctree.cytree as cytree
from ding.rl_utils.efficientzero.game import GameHistory
from ding.rl_utils.efficientzero.utils import select_action, prepare_observation_lst
from dizoo.board_games.atari.config.atari_config import game_config
from ding.rl_utils.efficientzero.mcts import MCTS

config = game_config

GameBuffer_config = EasyDict(dict(
    batch_size=10,
    transition_num=20,
    priority_prob_alpha=0.5,
    total_transitions=10000,
))
import argparse


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--save_video', action='store_true', default=False, help='save video in test.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=14, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int, default=20, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=8, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=4, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=80, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true', default=False,
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true', default=False,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=False, help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int, default=150 * 1024 * 1024 * 1024, help='object store memory')

    # Process arguments
    args = parser.parse_args()
    args.env = 'PongNoFrameskip-v4'
    args.case = 'atari'
    args.opr = 'train'
    args.amp_type = 'none'
    args.num_gpus = 0
    args.num_cpus = 16
    args.cpu_actor = 2
    args.gpu_actor = 0
    args.object_store_memory = 1 * 1024 * 1024 * 1024
    args.no_cuda = True
    args.device = 'cpu'
    # set config as per arguments
    exp_path = game_config.set_config(args)  # TODO

    done = False
    render = False
    save_video = False
    final_test = False
    use_pb = True
    counter = 0
    device = args.device
    test_episodes = 2
    config.max_moves = 20
    # to obtain model = EfficientZeroNet()
    model = config.get_uniform_network()
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))

    if use_pb:
        pb = tqdm(np.arange(config.max_moves), leave=True)

    with torch.no_grad():
        # new games
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                                video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]

        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(envs[_].env.action_space, max_length=config.max_moves, config=config) for
            _ in
            range(test_episodes)]
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
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            with autocast():
                # stack_obs {Tensor:(2,12,96,96)}
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state  # {ndarray:（2, 64, 6, 6）}
            reward_hidden_roots = network_output.reward_hidden  # {tuple:2}->{ndarray:(1,2,512)}
            value_prefix_pool = network_output.value_prefix   # {list: 2}->{float}
            policy_logits_pool = network_output.policy_logits.tolist() # {list: 2}->{list:6}->{float}

            roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()# {list: 1}->{list:6}
            roots_values = roots.get_values() # {list: 1}
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
                pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                   ''.format(config.env_name, counter,
                                             ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
                pb.update(1)

        for env in envs:
            env.close()
