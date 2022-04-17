"""The code is adapted from https://github.com/nikhilbarhate99/min-decision-transformer
"""
from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


class D4RLTrajectoryDataset(Dataset):

    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        if isinstance(self.trajectories[0], list):  # for cartpole
            self.trajectories_tmp = {}
            self.trajectories_tmp = [
                {
                    'observations': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['obs']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'next_observations': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['next_obs']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'actions': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['action']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'rewards': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['reward']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    # 'dones':
                    #     np.stack([
                    #     int(self.trajectories[eps_index][transition_index]['done']) for transition_index in range(len(self.trajectories[eps_index]))
                    # ], axis=0)
                } for eps_index in range(len(self.trajectories))
            ]

            self.trajectories = self.trajectories_tmp

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si:si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si:si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si:si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat(
                [states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], dim=0
            )

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat(
                [actions, torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)], dim=0
            )

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(([padding_len] + list(returns_to_go.shape[1:])), dtype=returns_to_go.dtype)
                ],
                dim=0
            )

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [torch.ones(traj_len, dtype=torch.long),
                 torch.zeros(padding_len, dtype=torch.long)], dim=0
            )

        return timesteps, states, actions, returns_to_go, traj_mask


def serial_pipeline_dt(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    traj_dataset = D4RLTrajectoryDataset(cfg.policy.learn.dataset_path, cfg.policy.context_len, cfg.policy.rtg_scale)
    traj_data_loader = DataLoader(
        traj_dataset, batch_size=cfg.policy.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    data_iter = iter(traj_data_loader)
    # get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False

    for i in range(max_train_iter):
        if i != 0 and i % 10 == 0:
            stop = policy.evaluate(state_mean, state_std)

        learner.train({'data_iter': data_iter, 'traj_data_loader': traj_data_loader})
        if stop:
            break
    learner.call_hook('after_run')
    return policy, stop
