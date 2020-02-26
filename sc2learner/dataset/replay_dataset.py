import os
import torch
import numpy as np
import numbers
import random
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from sc2learner.envs.observations.alphastar_obs_wrapper import decompress_obs


META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'


class ReplayDataset(Dataset):
    def __init__(self, cfg):
        super(ReplayDataset, self).__init__()
        assert(cfg.data.trajectory_type in ['random', 'slide_window'])
        with open(cfg.data.replay_list, 'r') as f:
            path_list = f.readlines()
        self.path_list = [{'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)]
        self.trajectory_len = cfg.data.trajectory_len
        self.trajectory_type = cfg.data.trajectory_type
        self.slide_window_step = cfg.data.slide_window_step
        self.use_stat = cfg.data.use_stat
        self.beginning_build_order_num = cfg.data.beginning_build_order_num
        self.beginning_build_order_prob = cfg.data.beginning_build_order_prob
        self.cumulative_stat_prob = cfg.data.cumulative_stat_prob
        self.use_global_cumulative_stat = cfg.data.use_global_cumulative_stat

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

    def step(self):
        for i in range(len(self.path_list)):
            handle = self.path_list[i]
            if 'step_num' not in handle.keys():
                meta = torch.load(handle['name'] + META_SUFFIX)
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

    def _load_stat(self, handle):
        stat = torch.load(handle['name'] + STAT_SUFFIX)
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
        data = torch.load(handle['name'] + DATA_SUFFIX)
        start = handle['cur_step']
        end = start + self.trajectory_len
        sample_data = data[start:end]
        sample_data = self.action_unit_id_transform(sample_data)
        # if unit id transform deletes some data frames,
        # collate_fn will use the minimum number of data frame to compose a batch
        sample_data = [decompress_obs(d) for d in sample_data]
        # check raw coordinate (x, y) <-> (y, x)
        assert(handle['map_size'] == list(reversed(sample_data[0]['spatial_info'].shape[1:])))
        if self.use_stat:
            beginning_build_order, cumulative_stat, mmr = self._load_stat(handle)
            for i in range(len(sample_data)):
                sample_data[i]['scalar_info']['beginning_build_order'] = beginning_build_order
                sample_data[i]['scalar_info']['mmr'] = mmr
                if self.use_global_cumulative_stat:
                    sample_data[i]['scalar_info']['cumulative_stat'] = cumulative_stat

        return sample_data


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


def policy_collate_fn(batch, max_delay=63):
    data_item = {
        'spatial_info': True,
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'actions': False
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
        new_data = list_dict2dict_list(data)
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
            if k == 'actions':
                new_data[k] = list_dict2dict_list(new_data[k])
                new_data[k]['delay'] = [torch.clamp(x, 0, max_delay) for x in new_data[k]['delay']]  # clip
        return new_data

    # sequence, batch
    seq = list(zip(*batch))
    for s in range(len(seq)):
        seq[s] = merge_func(seq[s])
    return seq
