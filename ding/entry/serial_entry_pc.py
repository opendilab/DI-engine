from typing import Union, Optional, Tuple
import os

import easydict
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import NaiveRLDataset

import numpy as np

from dizoo.maze.envs.maze_env import Maze


def print_obs(obs):
    print('Wall')
    print(obs[:, :, 0])
    print('Goal')
    print(obs[:, :, 1])
    print('Obs')
    print(obs[:, :, 2])


def get_vi_sequence(env, observation):
    """Returns [L, W, W] optimal actions."""
    xy = np.where(observation[Ellipsis, -1] == 1)
    start_x, start_y = xy[0][0], xy[1][0]
    target_location = env.target_location
    nav_map = env.nav_map
    current_points = [target_location]
    chosen_actions = {target_location: 0}
    visited_points = {target_location: True}
    vi_sequence = []
    vi_map = np.full((env.size, env.size), fill_value=env.n_action, dtype=np.int32)

    found_start = False
    while current_points and not found_start:
        next_points = []
        for point_x, point_y in current_points:
            for (action, (next_point_x, next_point_y)) in [(0, (point_x - 1, point_y)), (1, (point_x, point_y - 1)),
                                                           (2, (point_x + 1, point_y)), (3, (point_x, point_y + 1))]:

                if (next_point_x, next_point_y) in visited_points:
                    continue

                if not (0 <= next_point_x < len(nav_map) and 0 <= next_point_y < len(nav_map[next_point_x])):
                    continue

                if nav_map[next_point_x][next_point_y] == 'x':
                    continue

                next_points.append((next_point_x, next_point_y))
                visited_points[(next_point_x, next_point_y)] = True
                chosen_actions[(next_point_x, next_point_y)] = action
                vi_map[next_point_x, next_point_y] = action

                if next_point_x == start_x and next_point_y == start_y:
                    found_start = True
        vi_sequence.append(vi_map.copy())
        current_points = next_points

    return np.array(vi_sequence)


class PCDataset(Dataset):

    def __init__(self, all_data):
        self._data = all_data

    def __getitem__(self, item):
        return {'obs': self._data[0][item], 'bfs_in': self._data[1][item], 'bfs_out': self._data[2][item]}

    def __len__(self):
        return self._data[0].shape[0]


def load_2d_datasets(train_seeds=5, test_seeds=1, batch_size=32):

    def load_env(seed):
        ccc = easydict.EasyDict({'size': 16})
        e = Maze(ccc)
        e.seed(seed)
        e.reset()
        return e

    envs = [load_env(i) for i in range(train_seeds + test_seeds)]

    observations_train = []
    observations_test = []
    bfs_input_maps_train = []
    bfs_input_maps_test = []
    bfs_output_maps_train = []
    bfs_output_maps_test = []
    for idx, env in enumerate(envs):
        if idx < train_seeds:
            observations = observations_train
            bfs_input_maps = bfs_input_maps_train
            bfs_output_maps = bfs_output_maps_train
        else:
            observations = observations_test
            bfs_input_maps = bfs_input_maps_test
            bfs_output_maps = bfs_output_maps_test

        env_observations = torch.stack([torch.from_numpy(env.random_start()) for _ in range(80)])
        # assert False
        # env_observations = torch.squeeze(env_steps.observation, axis=1)
        for i in range(env_observations.shape[0]):
            bfs_sequence = get_vi_sequence(env, env_observations[i].numpy().astype(np.int32))  # [L, W, W]
            bfs_input_map = env.n_action * np.ones([env.size, env.size], dtype=np.long)
            for j in range(bfs_sequence.shape[0]):
                bfs_input_maps.append(torch.from_numpy(bfs_input_map))
                bfs_output_maps.append(torch.from_numpy(bfs_sequence[j]))
                observations.append(env_observations[i])
                bfs_input_map = bfs_sequence[j]

    train_data = PCDataset(
        (
            torch.stack(observations_train, dim=0),
            torch.stack(bfs_input_maps_train, dim=0),
            torch.stack(bfs_output_maps_train, dim=0),
        )
    )
    test_data = PCDataset(
        (
            torch.stack(observations_test, dim=0),
            torch.stack(bfs_input_maps_test, dim=0),
            torch.stack(bfs_output_maps_test, dim=0),
        )
    )

    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, test_dataset


def serial_pipeline_pc(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        max_iter=int(1e6),
) -> Union['Policy', bool]:  # noqa
    r"""
    Overview:
        Serial pipeline entry of imitation learning.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - data_path (:obj:`str`): Path of training data.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
        - convergence (:obj:`bool`): whether il training is converged
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Env, Policy
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    dataloader, test_dataloader = load_2d_datasets()
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    stop = False
    iter_cnt = 0
    for epoch in range(cfg.policy.learn.train_epoch):
        # Evaluate policy performance
        # loss_list = []
        # for _, bat in enumerate(eval_loader):
        #     res = policy._forward_eval(bat['obs'])
        #     if cont:
        #         loss_list.append(torch.nn.L1Loss()(res['action'], bat['action'].squeeze(-1)).item())
        #     else:
        #         res = torch.argmax(res['logit'], dim=1)
        #         loss_list.append(torch.sum(res == bat['action'].squeeze(-1)).item() / bat['action'].shape[0])
        # if cont:
        #     label = 'validation_loss'
        # else:
        #     label = 'validation_acc'
        # tb_logger.add_scalar(label, sum(loss_list) / len(loss_list), iter_cnt)

        # train
        criterion = torch.nn.CrossEntropyLoss()
        for i, train_data in enumerate(dataloader):
            learner.train(train_data)
            iter_cnt += 1
            if iter_cnt >= max_iter:
                stop = True
                break
        if stop:
            break
        losses = []
        acces = []
        for _, test_data in enumerate(test_dataloader):
            observations, bfs_input_maps, bfs_output_maps = test_data['obs'], test_data['bfs_in'].long(), \
                                                            test_data['bfs_out'].long()
            states = observations
            bfs_input_onehot = torch.nn.functional.one_hot(bfs_input_maps, 5).float()
            bfs_states = torch.cat([states, bfs_input_onehot], dim=-1).cuda()
            logits = policy._model(bfs_states)['logit']
            logits = logits.flatten(0, -2)
            labels = bfs_output_maps.flatten(0, -1).cuda()

            loss = criterion(logits, labels).item()
            preds = torch.argmax(logits, dim=-1)
            acc = torch.sum((preds == labels)) / preds.shape[0]

            losses.append(loss)
            acces.append(acc)
        print('Test Finished! Loss: {} acc: {}'.format(sum(losses) / len(losses), sum(acces) / len(acces)))
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
