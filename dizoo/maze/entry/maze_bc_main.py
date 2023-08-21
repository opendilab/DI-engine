from typing import Union, Optional, Tuple
import os
from functools import partial
from copy import deepcopy

import easydict
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from dizoo.maze.envs.maze_env import Maze


# BFS algorithm
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
    track_back = []
    if found_start:
        cur_x, cur_y = start_x, start_y
        while cur_x != target_location[0] or cur_y != target_location[1]:
            act = vi_sequence[-1][cur_x, cur_y]
            track_back.append((torch.FloatTensor(env.process_states([cur_x, cur_y], env.get_maze_map())), act))
            if act == 0:
                cur_x += 1
            elif act == 1:
                cur_y += 1
            elif act == 2:
                cur_x -= 1
            elif act == 3:
                cur_y -= 1

    return np.array(vi_sequence), track_back


class BCDataset(Dataset):

    def __init__(self, all_data):
        self._data = all_data

    def __getitem__(self, item):
        return {'obs': self._data[item][0], 'action': self._data[item][1]}

    def __len__(self):
        return len(self._data)


def load_bc_dataset(train_seeds=1, test_seeds=1, batch_size=32):

    def load_env(seed):
        ccc = easydict.EasyDict({'size': 16})
        e = Maze(ccc)
        e.seed(seed)
        e.reset()
        return e

    envs = [load_env(i) for i in range(train_seeds + test_seeds)]
    data_train = []
    data_test = []

    for idx, env in enumerate(envs):
        if idx < train_seeds:
            data = data_train
        else:
            data = data_test

        start_obs = env.process_states(env._get_obs(), env.get_maze_map())
        _, track_back = get_vi_sequence(env, start_obs)

        data += track_back

    train_data = BCDataset(data_train)
    test_data = BCDataset(data_test)

    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataset, test_dataset


def serial_pipeline_bc(
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
    dataloader, test_dataloader = load_bc_dataset()
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
        loss_list = []
        for _, bat in enumerate(test_dataloader):
            bat['action'] = bat['action'].long()
            res = policy._forward_eval(bat['obs'])
            res = torch.argmax(res['logit'], dim=1)
            loss_list.append(torch.sum(res == bat['action'].squeeze(-1)).item() / bat['action'].shape[0])
        label = 'validation_acc'
        tb_logger.add_scalar(label, sum(loss_list) / len(loss_list), iter_cnt)
        for i, train_data in enumerate(dataloader):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
                if stop:
                    break
            train_data['action'] = train_data['action'].long()
            learner.train(train_data)
            iter_cnt += 1
            if iter_cnt >= max_iter:
                stop = True
                break
        if stop:
            break

    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop


if __name__ == '__main__':
    from dizoo.maze.config.maze_bc_config import main_config, create_config
    serial_pipeline_bc([main_config, create_config], seed=0)
