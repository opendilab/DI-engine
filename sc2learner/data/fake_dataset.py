import random
import tempfile
from collections import OrderedDict
import copy

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
LSTM_DIMS = [3, 1, 384]

NOOP = None


def random_binary_tensor(size, dtype=torch.float32):
    return torch.randint(0, 1, size=size, dtype=dtype)


def random_tensor(size, dtype=torch.float32):
    return torch.randn(size=size, dtype=dtype)


def random_action_type():
    action_type = np.random.choice(NUM_ACTION_TYPES, [1])
    return torch.from_numpy(action_type).type(torch.int64)


def random_one_hot(size, dtype=torch.float32):
    assert len(size) == 1
    tensor = torch.zeros(size=size, dtype=dtype)
    tensor[np.random.choice(size[0])] = 1
    return tensor


def get_single_step_data():
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


class FakeReplayDataset(ReplayDataset):
    def __init__(self, cfg=None):
        # Completely independent with the config
        self.trajectory_len = cfg.get("trajectory_len", 11) if cfg else 11
        self.slide_window_step = cfg.get("slide_window_step", 1) if cfg else 1
        length = np.random.randint(10, 30)  # random number of path
        self.path_list = [dict(name=tempfile.mkstemp(), count=0) for _ in range(length)]

    def __getitem__(self, item):
        sample_batch = [self.get_single_step_data() for _ in range(self.trajectory_len)]
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


class FakeActorDataset:
    def __init__(self, trajectory_len=3):
        self.trajectory_len = trajectory_len

    def __len__(self):
        return 128  # pseudo length, only for implement interface

    def __getitem__(self, idx):
        return self.get_1v1_agent_data()

    def get_1v1_agent_data(self):
        def get_z():
            ret = {}
            ret['built_units'] = random_binary_tensor([120], dtype=torch.long)
            ret['effects'] = random_binary_tensor([83], dtype=torch.long)
            ret['upgrades'] = random_binary_tensor([60], dtype=torch.long)
            num = random.randint(10, 100)
            ret['build_order'] = {
                'type': torch.from_numpy(np.random.choice(range(NUM_ACTION_TYPES), size=num, replace=True)),
                'loc': torch.randint(*MAP_SIZE, size=(num, 2))
            }
            return ret

        def get_outputs():
            prob = np.random.random()
            ret = {}
            ret['action_type'] = torch.rand(NUM_ACTION_TYPES)
            ret['delay'] = torch.rand(1) * DELAY_MAX
            ret['queued'] = NOOP if np.random.random() > 0.8 else torch.randn(2)
            if prob < 0.5:
                ret['selected_units'] = NOOP
            else:
                num = random.randint(1, MAX_SELECTED_UNITS)
                ret['selected_units'] = torch.rand(num, NUM_UNIT_TYPES)
            if prob < 0.33:
                ret['target_units'] = NOOP
                ret['target_location'] = NOOP
            elif prob < 0.67:
                ret['target_units'] = torch.rand(NUM_UNIT_TYPES)
                ret['target_location'] = NOOP
            else:
                ret['target_units'] = NOOP
                ret['target_location'] = torch.rand(*MAP_SIZE)
            return ret

        def disturb_outputs(outputs):
            new_outputs = copy.deepcopy(outputs)
            new_outputs['action_type'] += torch.randn_like(new_outputs['action_type']) * 0.1
            new_outputs['delay'] = torch.clamp(new_outputs['delay'] + torch.randn(1) * 10, 0, DELAY_MAX)
            if new_outputs['queued'] != NOOP:
                new_outputs['queued'] += torch.randn_like(new_outputs['queued']) * 0.1
            if new_outputs['selected_units'] != NOOP:
                new_outputs['selected_units'] += torch.randn_like(new_outputs['selected_units']) * 0.1
            if new_outputs['target_units'] != NOOP:
                new_outputs['target_units'] += torch.randn_like(new_outputs['target_units']) * 0.1
            if new_outputs['target_location'] != NOOP:
                new_outputs['target_location'] += torch.randn_like(new_outputs['target_location']) * 0.1
            return new_outputs

        def get_single_rl_agent_step_data():
            base = get_single_step_data()
            base['prev_state'] = [torch.zeros(*LSTM_DIMS), torch.zeros(*LSTM_DIMS)]
            base['rewards'] = torch.randint(0, 1, size=[1])
            base['game_seconds'] = random.randint(0, 24 * 60)
            base['agent_z'] = get_z()
            base['target_z'] = get_z()
            base['target_outputs'] = get_outputs()
            base['behaviour_outputs'] = disturb_outputs(base['target_outputs']
                                                        ) if np.random.random() > 0.3 else get_outputs()
            base['teacher_outputs'] = disturb_outputs(base['target_outputs']
                                                      ) if np.random.random() > 0.3 else get_outputs()
            base['teacher_actions'] = copy.deepcopy(base['actions'])
            return base

        data = []
        for i in range(self.trajectory_len):
            data.append({'home': get_single_rl_agent_step_data(), 'away': get_single_rl_agent_step_data()})
        data[-1]['home_next'] = get_single_step_data()
        data[-1]['away_next'] = get_single_step_data()
        return data
