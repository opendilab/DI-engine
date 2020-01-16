import os
import torch
import random
from torch.utils.data import Dataset


META_SUFFIX = '.meta'
DATA_SUFFIX = '.data'


class ReplayDataset(Dataset):
    def __init__(self, replay_list, trajectory_len=64, trajectory_type='random', slide_window_step=1):
        super(ReplayDataset, self).__init__()
        assert(trajectory_type in ['random', 'slide_window'])
        with open(replay_list, 'r') as f:
            path_list = f.readlines()
        # need to be added into checkpoint
        self.path_dict = {idx: {'name': p, 'count': 0} for idx, p in enumerate(path_list)}
        self.trajectory_len = trajectory_len
        self.trajectory_type = trajectory_type
        self.slide_window_step = slide_window_step

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
        step_num = self._get_item_step_num(idx)
        if self.trajectory_type == 'random':
            start = random.randint(0, step_num - self.trajectory_len)
        elif self.trajectory_type == 'slide_window_step':
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
        return data[start:end]


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
    selected_replay = '\n'.join(selected_replay)
    with open(output_path, 'w') as f:
        f.writelines(selected_replay)
