import random
import tempfile
from collections import OrderedDict

import numpy as np
import torch

from pysc2.lib.static_data import ACTIONS_REORDER, NUM_UNIT_TYPES
from sc2learner.data.offline.replay_dataset import ReplayDataset, START_STEP

META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'
MAP_SIZE = [176, 200]
DELAY_MAX = 63
MAX_SELECTED_UNITS = 64
ACTION_CANDIDATES = list(ACTIONS_REORDER.keys())
NUM_ACTION_TYPES = len(ACTION_CANDIDATES)

NOOP = "none"


def random_binary_tensor(size, dtype=torch.float32):
    return torch.randint(0, 1, size=size, dtype=dtype)


def random_tensor(size, dtype=torch.float32):
    return torch.randn(size=size, dtype=dtype)


def random_action_type():
    action_type = np.random.choice(ACTION_CANDIDATES, [1])
    return torch.from_numpy(action_type).type(torch.int64)


def random_one_hot(size, dtype=torch.float32):
    assert len(size) == 1
    tensor = torch.zeros(size=size, dtype=dtype)
    tensor[np.random.choice(size[0])] = 1
    return tensor


class FakeReplayDataset(ReplayDataset):
    def __init__(self, cfg=None):
        # Completely independent with the config
        self.trajectory_len = cfg.get("trajectory_len", 11) if cfg else 11
        self.slide_window_step = cfg.get("slide_window_step", 1) if cfg else 1
        length = np.random.randint(10, 30)  # random number of path
        self.path_list = [dict(name=tempfile.mkstemp(), count=0) for _ in range(length)]

    def __getitem__(self, item):
        sample_batch = [self.__get_single_step_data() for _ in range(self.trajectory_len)]
        sample_batch[0][START_STEP] = np.random.random() > 0.5
        return sample_batch

    def step(self, index=None):
        if index is None:
            index = range(len(self.path_list))
        end_list = []  # a list contains replay index which access the end of the replay
        for i in index:
            handle = self.path_list[i]
            if 'step_num' not in handle.keys():
                meta = {
                    "step_num": 100,  # FIXME Not sure what this value should be
                    "map_size": [176, 200]
                }
                step_num = meta['step_num']
                handle['step_num'] = step_num
                handle['map_size'] = meta['map_size']
            else:
                step_num = handle['step_num']
            assert (handle['step_num'] >= self.trajectory_len)
            # if self.trajectory_type == 'random':
            # handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
            # elif self.trajectory_type == 'slide_window':
            if 'cur_step' not in handle.keys():
                handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
            else:
                next_step = handle['cur_step'] + self.slide_window_step
                if next_step >= step_num - self.trajectory_len:
                    handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                else:
                    handle['cur_step'] = next_step
            # elif self.trajectory_type == 'sequential':
            # if 'cur_step' not in handle.keys():
            #     handle['cur_step'] = 0
            # else:
            # next_step = handle['cur_step'] + self.slide_window_step
            # if next_step >= step_num - self.trajectory_len:
            #     end_list.append(i)
            # handle['cur_step'] = next_step
        return end_list

    def reset_step(self, index=None):
        if index is None:
            index = range(len(self.path_list))
        for i in index:
            self.path_list[i].pop('cur_step')

    def _load_stat(self, handle=None):
        beginning_build_order = random_binary_tensor([20, 283])
        cumulative_stat = OrderedDict(
            unit_build=random_binary_tensor([120]),
            effect=random_binary_tensor([83]),
            research=random_binary_tensor([60])
        )
        mmr = random_binary_tensor([7])
        return beginning_build_order, cumulative_stat, mmr

    def __get_single_step_data(self):
        # TODO(pzh) we should build a general data structure (a SampleBatch class)

        # TODO(pzh) our system should not high dependent on the data structure, because we may add / delete item
        #  in future

        num_units = np.random.randint(200, 300)
        selected_num_units = np.random.randint(1, MAX_SELECTED_UNITS)

        # TODO(pzh) we should use a more light-weight data type to store binary.
        scalar_info = OrderedDict(
            agent_statistics=random_tensor([10]),
            race=random_one_hot([5]),
            enemy_race=random_one_hot([5]),
            upgrades=random_binary_tensor([90]),
            enemy_upgrades=random_binary_tensor([90]),
            time=random_binary_tensor([32]),
            available_actions=random_binary_tensor([NUM_ACTION_TYPES]),
            unit_counts_bow=random_tensor([259]),
            last_delay=random_binary_tensor([6]),
            last_queued=random_binary_tensor([3]),
            last_action_type=random_one_hot([NUM_ACTION_TYPES]),
            mmr=random_binary_tensor([7]),
            cumulative_stat=OrderedDict(
                unit_build=random_binary_tensor([120]),
                effect=random_binary_tensor([83]),
                research=random_binary_tensor([60])
            ),
            beginning_build_order=random_tensor([20, 283]),
        )

        entity_raw = OrderedDict(
            location=list(np.random.randint(0, min(MAP_SIZE), size=[num_units, 2], dtype=int)),
            id=list(range(8600000000, 8600000000 + num_units)),
            type=list(np.random.randint(0, NUM_UNIT_TYPES, size=num_units, dtype=int))
        )

        # TODO(pzh) it's all int64 here. not correct.

        actions = OrderedDict(
            action_type=random_action_type(),
            delay=torch.randint(0, DELAY_MAX, size=[1], dtype=torch.int64),
            queued=NOOP if np.random.random() > 0.8 else torch.randint(0, 1, size=[1], dtype=torch.int64),
            selected_units=NOOP
            if np.random.random() > 0.8 else torch.randint(0, num_units, size=[selected_num_units], dtype=torch.int64),
            target_units=NOOP if np.random.random() > 0.8 else torch.randint(0, num_units, size=[1], dtype=torch.int64),
            target_location=NOOP
            if np.random.random() > 0.8 else torch.randint(0, min(MAP_SIZE), size=[2], dtype=torch.int64)
        )
        return OrderedDict(
            scalar_info=scalar_info,
            entity_raw=entity_raw,
            actions=actions,
            entity_info=random_tensor([num_units, 2102]),
            spatial_info=random_tensor([20] + MAP_SIZE),
            map_size=MAP_SIZE,
        )
