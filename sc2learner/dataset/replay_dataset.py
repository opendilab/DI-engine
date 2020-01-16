import os
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'


class ReplayDataset(Dataset):
    def __init__(self, replay_list, trajectory_len=64, trajectory_type='random', slide_window_step=1,
                 data_type='only_policy'):
        super(ReplayDataset, self).__init__()
        assert(trajectory_type in ['random', 'slide_window'])
        assert(data_type in ['only_policy', 'total'])
        with open(replay_list, 'r') as f:
            path_list = f.readlines()
        # need to be added into checkpoint
        self.path_dict = {idx: {'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)}
        self.trajectory_len = trajectory_len
        self.trajectory_type = trajectory_type
        self.slide_window_step = slide_window_step
        self.data_type = data_type

    def __len__(self):
        return len(self.path_dict.keys())

    def state_dict(self):
        return self.path_dict

    def _get_item_step_num(self, handle, idx):
        if 'step_num' in handle.keys():
            print('enter in _get_item_step_num')  # TODO validate
            return handle['step_num']
        else:
            meta = torch.load(handle['name'] + META_SUFFIX)
            step_num = meta['step_num']
            handle['step_num'] = step_num
            return step_num

    def __getitem__(self, idx):
        handle = self.path_dict[idx]
        data = torch.load(handle['name'] + DATA_SUFFIX)
        step_num = self._get_item_step_num(handle, idx)
        if self.trajectory_type == 'random':
            start = random.randint(0, step_num - self.trajectory_len)
        elif self.trajectory_type == 'slide_window':
            if 'cur_step' in handle.keys():
                start = handle['cur_step']
            else:
                start = random.randint(0, step_num - self.trajectory_len)
                handle['cur_step'] = start
            next_step = handle['cur_step'] + self.slide_window_step
            if next_step >= step_num - self.trajectory_len:
                handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
            else:
                handle['cur_step'] = next_step
        end = start + self.trajectory_len
        handle['count'] += 1
        sample_data = data[start:end]
        if self.data_type == 'only_policy':
            for i in range(len(sample_data)):
                temp = sample_data[i]['obs0']
                temp['actions'] = sample_data[i]['act']
                sample_data[i] = temp
        return sample_data


def select_replay(replay_dir, min_mmr=0, home_race=None, away_race=None):
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
            selected_replay.append(os.path.join(replay_dir, name))
    return selected_replay


def get_replay_list(replay_dir, output_path, **kwargs):
    selected_replay = select_replay(replay_dir, **kwargs)
    selected_replay = [p+'\n' for p in selected_replay]
    with open(output_path, 'w') as f:
        f.writelines(selected_replay)


def policy_collate_fn(batch):
    data_item = {
        'spatial_info': True,
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'actions': False
    }

    def merge_func(data):
        new_data = {k: [] for k in data_item}
        for b in range(len(data)):
            for k in data_item.keys():
                new_data[k].append(data[b][k])
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
        return new_data

    # sequence, batch
    seq = list(zip(*batch))
    for s in range(len(seq)):
        seq[s] = merge_func(seq[s])
    return seq
