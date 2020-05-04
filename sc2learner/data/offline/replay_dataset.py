import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset

from pysc2.lib.static_data import ACTIONS_REORDER, NUM_UPGRADES
from sc2learner.data.base_dataset import BaseDataset
from sc2learner.envs import get_available_actions_processed_data, decompress_obs, action_unit_id_transform,\
    get_enemy_upgrades_processed_data
from sc2learner.utils import read_file_ceph, list_dict2dict_list

META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'

START_STEP = "start_step"


class ReplayDataset(BaseDataset):
    def __init__(self, dataset_config, train_mode=True):
        super(ReplayDataset, self).__init__(dataset_config)
        with open(dataset_config.replay_list, 'r') as f:
            path_list = f.readlines()
        self.path_list = [{'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)]

        # if train_mode is set to True, then we return a clipped version of data. Otherwise return the whole episode.
        self.complete_episode = not train_mode

        if not self.complete_episode:
            assert dataset_config.trajectory_type in ['random', 'slide_window', 'sequential'], dataset_config
            self.trajectory_len = dataset_config.trajectory_len
            self.trajectory_type = dataset_config.trajectory_type
            self.slide_window_step = dataset_config.slide_window_step
            self.beginning_build_order_prob = dataset_config.beginning_build_order_prob
            self.cumulative_stat_prob = dataset_config.cumulative_stat_prob

        self.use_stat = dataset_config.use_stat
        self.beginning_build_order_num = dataset_config.beginning_build_order_num
        self.use_global_cumulative_stat = dataset_config.use_global_cumulative_stat
        self.use_ceph = dataset_config.use_ceph
        self.use_available_action_transform = dataset_config.use_available_action_transform
        self.use_enemy_upgrades = dataset_config.use_enemy_upgrades

    def __len__(self):
        return len(self.path_list)

    def state_dict(self):
        return self.path_list

    def load_state_dict(self, state_dict):
        self.path_list = state_dict

    def step(self, index=None):
        assert not self.complete_episode, "During evaluation, we don't need to step the dataset."
        if index is None:
            index = range(len(self.path_list))
        end_list = []  # a list contains replay index which access the end of the replay
        for i in index:
            handle = self.path_list[i]
            if 'step_num' not in handle.keys():
                meta = torch.load(self._read_file(handle['name'] + META_SUFFIX))
                step_num = meta['step_num']
                handle['step_num'] = step_num
                handle['map_size'] = meta['map_size']
            else:
                step_num = handle['step_num']
            assert (handle['step_num'] >= self.trajectory_len)
            if self.trajectory_type == 'random':
                handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
            elif self.trajectory_type == 'slide_window':
                if 'cur_step' not in handle.keys():
                    handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                else:
                    next_step = handle['cur_step'] + self.slide_window_step
                    if next_step >= step_num - self.trajectory_len:
                        handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                    else:
                        handle['cur_step'] = next_step
            elif self.trajectory_type == 'sequential':
                if 'cur_step' not in handle.keys():
                    handle['cur_step'] = 0
                else:
                    next_step = handle['cur_step'] + self.slide_window_step
                    if next_step >= step_num - self.trajectory_len:
                        end_list.append(i)
                    handle['cur_step'] = next_step
        return end_list

    def reset_step(self, index=None):
        if index is None:
            index = range(len(self.path_list))
        for i in index:
            self.path_list[i].pop('cur_step')

    def _read_file(self, path, read_type='BytesIO'):
        if self.use_ceph:
            return read_file_ceph(path, read_type=read_type)
        else:
            return path

    def _load_stat(self, handle):
        stat = torch.load(self._read_file(handle['name'] + STAT_SUFFIX))
        mmr = stat['mmr']
        beginning_build_order = stat['beginning_build_order']
        # first self.beginning_build_order_num item
        beginning_build_order = beginning_build_order[:self.beginning_build_order_num]
        if beginning_build_order.shape[0] < self.beginning_build_order_num:
            B, N = beginning_build_order.shape
            B0 = self.beginning_build_order_num - B
            beginning_build_order = torch.cat([beginning_build_order, torch.zeros(B0, N)])
        cumulative_stat = stat['cumulative_stat']
        return beginning_build_order, cumulative_stat, mmr

    def __getitem__(self, idx):
        handle = self.path_list[idx]
        print(handle)

        d1 = self._read_file(handle['name'] + DATA_SUFFIX)
        data = torch.load(d1)

        # clip the dataset
        if self.complete_episode:
            start = 0
            sample_data = data
            self.bool_bo, self.bool_cum = 1, 1
        else:
            start = handle['cur_step']
            end = start + self.trajectory_len
            sample_data = data[start:end]
            self.bool_bo = float(np.random.rand() < self.beginning_build_order_prob)
            self.bool_cum = float(np.random.rand() < self.cumulative_stat_prob)

        sample_data = action_unit_id_transform(sample_data)
        sample_data = [decompress_obs(d) for d in sample_data]
        if self.use_available_action_transform:
            sample_data = [get_available_actions_processed_data(d) for d in sample_data]

        if self.complete_episode:
            meta = torch.load(self._read_file(handle['name'] + META_SUFFIX))
            map_size = list(reversed(meta['map_size']))
        else:
            # check raw coordinate (x, y) <-> (y, x)
            try:
                assert handle['map_size'] == list(reversed(sample_data[0]['spatial_info'].shape[1:]))
            except AssertionError as e:
                print('[Error] data name: {}'.format(handle['name']))
                raise e
            map_size = list(reversed(handle['map_size']))

        if self.use_stat:
            beginning_build_order, cumulative_stat, mmr = self._load_stat(handle)

        enemy_upgrades = None
        bool_cum = float(np.random.rand() < self.cumulative_stat_prob)
        for i in range(len(sample_data)):
            sample_data[i]['map_size'] = map_size
            if self.use_stat:
                sample_data[i]['scalar_info']['beginning_build_order'] = beginning_build_order * self.bool_bo
                sample_data[i]['scalar_info']['mmr'] = mmr
                if self.use_global_cumulative_stat:
                    step_cum_stat = cumulative_stat
                else:
                    step_cum_stat = sample_data[i]['scalar_info']['cumulative_stat']
                sample_data[i]['scalar_info']['cumulative_stat'] = {
                    k: self.bool_cum * v
                    for k, v in step_cum_stat.items()
                }

            if self.use_enemy_upgrades:
                enemy_upgrades = get_enemy_upgrades_processed_data(sample_data[i], enemy_upgrades)
                sample_data[i]['scalar_info']['enemy_upgrades'] = enemy_upgrades.float()

        if start == 0:
            sample_data[0][START_STEP] = True
        else:
            sample_data[0][START_STEP] = False
        return sample_data
