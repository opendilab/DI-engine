import os
import torch
import torch.nn.functional as F
import numpy as np
import numbers
import random
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from sc2learner.envs import get_available_actions_processed_data, decompress_obs
from sc2learner.utils import read_file_ceph
from pysc2.lib.static_data import ACTIONS_REORDER


META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'


class ReplayDataset(Dataset):
    def __init__(self, cfg):
        super(ReplayDataset, self).__init__()
        assert(cfg.trajectory_type in ['random', 'slide_window', 'sequential'])
        with open(cfg.replay_list, 'r') as f:
            path_list = f.readlines()
        self.path_list = [{'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)]
        self.trajectory_len = cfg.trajectory_len
        self.trajectory_type = cfg.trajectory_type
        self.slide_window_step = cfg.slide_window_step
        self.use_stat = cfg.use_stat
        self.beginning_build_order_num = cfg.beginning_build_order_num
        self.beginning_build_order_prob = cfg.beginning_build_order_prob
        self.cumulative_stat_prob = cfg.cumulative_stat_prob
        self.use_global_cumulative_stat = cfg.use_global_cumulative_stat
        self.use_ceph = cfg.use_ceph
        self.use_available_action_transform = cfg.use_available_action_transform

    def __len__(self):
        return len(self.path_list)

    def state_dict(self):
        return self.path_list

    def load_state_dict(self, state_dict):
        self.path_list = state_dict

    def copy(self, data):
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_data[k] = self.copy(v)
        elif isinstance(data, list) or isinstance(data, tuple):
            new_data = []
            for item in data:
                new_data.append(self.copy(item))
        elif isinstance(data, torch.Tensor):
            new_data = data.clone()
        elif isinstance(data, np.ndarray):
            new_data = np.copy(data)
        elif isinstance(data, str) or isinstance(data, numbers.Integral):
            new_data = data
        else:
            raise TypeError("invalid data type:{}".format(type(data)))
        return new_data

    def action_unit_id_transform(self, data):
        new_data = []
        for idx, item in enumerate(data):
            valid = True
            item = self.copy(data[idx])
            id_list = item['entity_raw']['id']
            action = item['actions']
            if isinstance(action['selected_units'], torch.Tensor):
                unit_ids = []
                for unit in action['selected_units']:
                    val = unit.item()
                    if val in id_list:
                        unit_ids.append(id_list.index(val))
                    else:
                        print("not found selected_units id({}) in nearest observation".format(val))
                        valid = False
                        break
                item['actions']['selected_units'] = torch.LongTensor(unit_ids)
            if isinstance(action['target_units'], torch.Tensor):
                unit_ids = []
                for unit in action['target_units']:
                    val = unit.item()
                    if val in id_list:
                        unit_ids.append(id_list.index(val))
                    else:
                        print("not found target_units id({}) in nearest observation".format(val))
                        valid = False
                        break
                item['actions']['target_units'] = torch.LongTensor(unit_ids)
            if valid:
                new_data.append(item)
        return new_data

    def step(self, index=None):
        if index is None:
            index = range(len(self.path_list))
        end_list = []  # a list containes replay index which access the end of the replay
        for i in index:
            handle = self.path_list[i]
            if 'step_num' not in handle.keys():
                meta = torch.load(self._read_file(handle['name'] + META_SUFFIX))
                step_num = meta['step_num']
                handle['step_num'] = step_num
                handle['map_size'] = meta['map_size']
            else:
                step_num = handle['step_num']
            assert(handle['step_num'] >= self.trajectory_len)
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

    def _read_file(self, path):
        if self.use_ceph:
            return read_file_ceph(path)
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
        bool_bo = float(np.random.rand() < self.beginning_build_order_prob)
        bool_cum = float(np.random.rand() < self.cumulative_stat_prob)
        beginning_build_order = bool_bo * beginning_build_order
        cumulative_stat = {k: bool_cum * v for k, v in cumulative_stat.items()}
        return beginning_build_order, cumulative_stat, mmr

    def __getitem__(self, idx):
        handle = self.path_list[idx]
        data = torch.load(self._read_file(handle['name'] + DATA_SUFFIX))
        start = handle['cur_step']
        end = start + self.trajectory_len
        sample_data = data[start:end]
        sample_data = self.action_unit_id_transform(sample_data)
        # if unit id transform deletes some data frames,
        # collate_fn will use the minimum number of data frame to compose a batch
        sample_data = [decompress_obs(d) for d in sample_data]
        if self.use_available_action_transform:
            sample_data = [get_available_actions_processed_data(d) for d in sample_data]
        # check raw coordinate (x, y) <-> (y, x)
        try:
            assert(handle['map_size'] == list(reversed(sample_data[0]['spatial_info'].shape[1:])))
        except AssertionError as e:
            print('[Error] data name: {}'.format(handle['name']))
            raise e
        map_size = list(reversed(handle['map_size']))
        if self.use_stat:
            beginning_build_order, cumulative_stat, mmr = self._load_stat(handle)
        for i in range(len(sample_data)):
            sample_data[i]['map_size'] = map_size
            if self.use_stat:
                sample_data[i]['scalar_info']['beginning_build_order'] = beginning_build_order
                sample_data[i]['scalar_info']['mmr'] = mmr
                if self.use_global_cumulative_stat:
                    sample_data[i]['scalar_info']['cumulative_stat'] = cumulative_stat
        if start == 0:
            sample_data[0]['start_step'] = True
        else:
            sample_data[0]['start_step'] = False

        return sample_data


class ReplayEvalDataset(ReplayDataset):
    def __init__(self, cfg):
        with open(cfg.replay_list, 'r') as f:
            path_list = f.readlines()
        self.path_list = [{'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)]
        self.use_stat = cfg.use_stat
        self.beginning_build_order_num = cfg.beginning_build_order_num
        self.use_global_cumulative_stat = cfg.use_global_cumulative_stat
        self.use_ceph = cfg.use_ceph
        self.use_available_action_transform = cfg.use_available_action_transform

    # overwrite
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

    # overwrite
    def __getitem__(self, idx):
        handle = self.path_list[idx]
        data = torch.load(self._read_file(handle['name'] + DATA_SUFFIX))
        data = self.action_unit_id_transform(data)
        data = [decompress_obs(d) for d in data]
        if self.use_available_action_transform:
            data = [get_available_actions_processed_data(d) for d in data]
        meta = torch.load(self._read_file(handle['name'] + META_SUFFIX))
        map_size = list(reversed(meta['map_size']))
        if self.use_stat:
            beginning_build_order, cumulative_stat, mmr = self._load_stat(handle)
        for i in range(len(data)):
            data[i]['map_size'] = map_size
            if self.use_stat:
                data[i]['scalar_info']['beginning_build_order'] = beginning_build_order
                data[i]['scalar_info']['mmr'] = mmr
                if self.use_global_cumulative_stat:
                    data[i]['scalar_info']['cumulative_stat'] = cumulative_stat

        return data


def select_replay(replay_dir, min_mmr=0, home_race=None, away_race=None, trajectory_len=64):
    race_list = ['Protoss', 'Terran', 'Zerg']
    assert(home_race is None or home_race in race_list)
    assert(away_race is None or away_race in race_list)
    selected_replay = []
    for item in os.listdir(replay_dir):
        name, suffix = item.split('.')
        if suffix == META_SUFFIX[1:]:
            home, away, mmr = name.split('_')[:3]
            if int(mmr) < min_mmr:
                continue
            if home_race and home != home_race:
                continue
            if away_race and away != away_race:
                continue
            meta = torch.load(os.path.join(replay_dir, name)+META_SUFFIX)
            if meta['step_num'] < trajectory_len:
                continue
            selected_replay.append(os.path.join(replay_dir, name))
    return selected_replay


def get_replay_list(replay_dir, output_path, **kwargs):
    selected_replay = select_replay(replay_dir, **kwargs)
    selected_replay = [p+'\n' for p in selected_replay]
    with open(output_path, 'w') as f:
        f.writelines(selected_replay)


def policy_collate_fn(batch, max_delay=63, action_type_transform=True):
    data_item = {
        'spatial_info': False,  # special op
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'actions': False,
        'map_size': False,
        'start_step': False
    }

    def list_dict2dict_list(data):
        if len(data) == 0:
            raise ValueError("empty data")
        keys = data[0].keys()
        new_data = {k: [] for k in keys}
        for b in range(len(data)):
            for k in keys:
                new_data[k].append(data[b][k])
        return new_data

    def merge_func(data):
        valid_data = [t for t in data if t is not None]
        new_data = list_dict2dict_list(valid_data)
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
            if k == 'spatial_info':
                shape = [t.shape for t in new_data[k]]
                if len(set(shape)) != 1:
                    tmp_shape = list(zip(*shape))
                    H, W = max(tmp_shape[1]), max(tmp_shape[2])
                    new_spatial_info = []
                    for item in new_data[k]:
                        h, w = item.shape[-2:]
                        new_spatial_info.append(F.pad(item, [0, W-w, 0, H-h], "constant", 0))
                    new_data[k] = default_collate(new_spatial_info)
                else:
                    new_data[k] = default_collate(new_data[k])
            if k == 'actions':
                new_data[k] = list_dict2dict_list(new_data[k])
                new_data[k]['delay'] = [torch.clamp(x, 0, max_delay) for x in new_data[k]['delay']]  # clip
                if action_type_transform:
                    action_type = [t.item() for t in new_data[k]['action_type']]
                    L = len(action_type)
                    for i in range(L):
                        action_type[i] = ACTIONS_REORDER[action_type[i]]
                    action_type = torch.LongTensor(action_type)
                    new_data[k]['action_type'] = list(torch.chunk(action_type, L, dim=0))
        new_data['end_index'] = [idx for idx, t in enumerate(data) if t is None]
        return new_data

    # sequence, batch
    b_len = [len(b) for b in batch]
    max_len = max(b_len)
    min_len = min(b_len)
    if max_len == min_len:
        seq = list(zip(*batch))
    else:
        seq = []
        for i in range(max_len):
            tmp = []
            for j in range(len(batch)):
                if i >= b_len[j]:
                    tmp.append(None)
                else:
                    tmp.append(batch[j][i])
            seq.append(tmp)

    seq = list(zip(*batch))
    for s in range(len(seq)):
        seq[s] = merge_func(seq[s])
    return seq
