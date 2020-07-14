import random
import tempfile
from collections import OrderedDict
import copy
import os
import numpy as np
import torch

from pysc2.lib.static_data import ACTIONS_REORDER, NUM_UNIT_TYPES, ACTIONS_REORDER_INV, NUM_BEGIN_ACTIONS, NUM_UPGRADES
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from sc2learner.data.offline.replay_dataset import START_STEP
from sc2learner.utils import get_step_data_compressor
from sc2learner.envs.observation import LOCATION_BIT_NUM

ENTITY_INFO_DIM = 1340
META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'
MAP_SIZE = [176, 200]
DELAY_MAX = 127
MAX_SELECTED_UNITS = 64
ACTION_CANDIDATES = list(ACTIONS_REORDER.values())
NUM_ACTION_TYPES = len(ACTION_CANDIDATES)
LSTM_DIMS = [3, 1, 384]

NOOP = None


def random_binary_tensor(size, dtype=torch.float32):
    randn = torch.randn(size)
    return torch.where(randn > 0, torch.ones_like(randn), torch.zeros_like(randn)).to(dtype)


def random_tensor(size, dtype=torch.float32):
    return torch.randn(size=size, dtype=dtype)


def random_action_type():
    action_type_idx = np.random.choice(range(len(ACTION_CANDIDATES)))
    action_type = ACTION_CANDIDATES[action_type_idx]
    return torch.LongTensor([action_type])


def random_one_hot(size, dtype=torch.float32):
    assert len(size) == 1
    tensor = torch.zeros(size=size, dtype=dtype)
    tensor[np.random.choice(size[0])] = 1
    return tensor


def get_fake_rewards():
    rewards = {}
    rewards['winloss'] = torch.randint(-1, 2, size=(1, ))
    rewards['build_order'] = torch.randint(-20, 1, size=(1, ))
    rewards['built_unit'] = torch.randint(-10, 1, size=(1, ))
    rewards['upgrade'] = torch.randint(-10, 1, size=(1, ))
    rewards['effect'] = torch.randint(-10, 1, size=(1, ))
    rewards['battle'] = torch.randint(0, 100000, size=[1])
    for k in rewards.keys():
        rewards[k] = rewards[k].float()
    return rewards


def get_random_action(num_units=None, selected_num_units=None):
    if num_units is None:
        num_units = np.random.randint(200, 300)
    if selected_num_units is None:
        selected_num_units = np.random.randint(1, MAX_SELECTED_UNITS)
    action_type = random_action_type()
    action_type_inv = ACTIONS_REORDER_INV[action_type.item()]
    action_attr = GENERAL_ACTION_INFO_MASK[action_type_inv]
    action = OrderedDict(
        action_type=action_type,
        delay=torch.randint(0, 4, size=[1], dtype=torch.int64),  # 4 for convenience
        queued=torch.randint(0, 1, size=[1], dtype=torch.int64) if action_attr['queued'] else NOOP,
        selected_units=torch.randint(0, num_units, size=[selected_num_units], dtype=torch.int64)
        if action_attr['selected_units'] else NOOP,
        target_units=torch.randint(0, num_units, size=[1], dtype=torch.int64) if action_attr['target_units'] else NOOP,
        target_location=torch.randint(0, min(MAP_SIZE), size=[2], dtype=torch.int64)
        if action_attr['target_location'] else NOOP
    )
    return action


def get_scalar_encoder_data():
    scalar_info = OrderedDict(
        agent_statistics=random_tensor([10]),
        race=random_one_hot([5]),
        enemy_race=random_one_hot([5]),
        upgrades=random_binary_tensor([NUM_UPGRADES]),
        enemy_upgrades=random_binary_tensor([48]),  # refer to envs/observations/enemy_upgrades.py
        time=random_binary_tensor([64]),
        available_actions=random_binary_tensor([NUM_ACTION_TYPES]),
        unit_counts_bow=random_tensor([259]),
        last_delay=random_one_hot([128]),
        last_queued=random_binary_tensor([20]),
        last_action_type=random_one_hot([NUM_ACTION_TYPES]),
        mmr=random_binary_tensor([7]),
        cumulative_stat=OrderedDict(
            unit_build=random_binary_tensor([120]),
            effect=random_binary_tensor([83]),
            research=random_binary_tensor([60])
        ),
        beginning_build_order=random_tensor([20, NUM_BEGIN_ACTIONS + 2 * LOCATION_BIT_NUM]),
    )
    return scalar_info


def get_single_step_data():
    # TODO(pzh) we should build a general data structure (a SampleBatch class)

    # TODO(pzh) our system should not high dependent on the data structure, because we may add / delete item
    #  in future

    num_units = np.random.randint(200, 300)
    selected_num_units = np.random.randint(1, MAX_SELECTED_UNITS)
    scalar_info = get_scalar_encoder_data()

    entity_raw = OrderedDict(
        location=list(np.random.randint(0, min(MAP_SIZE), size=[num_units, 2], dtype=int)),
        id=list(range(8600000000, 8600000000 + num_units)),
        type=list(np.random.randint(0, NUM_UNIT_TYPES, size=num_units, dtype=int))
    )

    action = get_random_action(num_units, selected_num_units)

    scalar_info['available_actions'][action['action_type']] = 1

    return OrderedDict(
        scalar_info=scalar_info,
        entity_raw=entity_raw,
        actions=action,
        entity_info=random_tensor([num_units, ENTITY_INFO_DIM]),
        spatial_info=random_tensor([20] + MAP_SIZE),
        map_size=MAP_SIZE,
    )


def fake_stat_processed():
    # produced by base64.b64encode(zlib.compress(pickle.dumps(torch.load(
    # r'Zerg_Terran_2479_0005e0d00cf4bca92d0432ecf23bd227337e39f7f8870a560ac3abfe7f89abc1.stat_processed'))))
    import pickle
    import base64
    import zlib
    path = os.path.join(
        os.path.dirname(__file__),
        'stat_data/Zerg_Terran_2479_0005e0d00cf4bca92d0432ecf23bd227337e39f7f8870a560ac3abfe7f89abc1.stat_processed_new'
    )
    example_stat_processed = base64.b64encode(zlib.compress(pickle.dumps(torch.load(path))))
    example_stat_processed = pickle.loads(zlib.decompress(base64.b64decode(example_stat_processed)))
    return example_stat_processed


def fake_stat_processed_professional_player():
    import pickle
    import base64
    import zlib
    path = os.path.join(os.path.dirname(__file__), 'stat_data/KJ_2498_win.z')
    example_stat_processed = base64.b64encode(zlib.compress(pickle.dumps(torch.load(path))))
    example_stat_processed = pickle.loads(zlib.decompress(base64.b64decode(example_stat_processed)))
    return example_stat_processed


def get_z():
    ret = {}
    ret['built_unit'] = random_binary_tensor([120], dtype=torch.long)
    ret['effect'] = random_binary_tensor([83], dtype=torch.long)
    ret['upgrade'] = random_binary_tensor([60], dtype=torch.long)
    num = random.randint(10, 100)
    ret['build_order'] = {
        'type': torch.from_numpy(np.random.choice(range(NUM_ACTION_TYPES), size=num, replace=True)).long(),
        'loc': torch.randint(*MAP_SIZE, size=(num, 2)).long()
    }
    return ret


class FakeReplayDataset:
    def __init__(self, cfg=None):
        # Completely independent with the config
        self.trajectory_len = cfg.get("trajectory_len", 11) if cfg else 11
        self.slide_window_step = cfg.get("slide_window_step", 1) if cfg else 1
        length = np.random.randint(10, 30)  # random number of path
        self.path_list = [dict(name=tempfile.mkstemp(), count=0) for _ in range(length)]

    def __getitem__(self, item):
        sample_batch = [get_single_step_data() for _ in range(self.trajectory_len)]
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

    def __len__(self):
        return 100


class FakeActorDataset:
    def __init__(self, cfg=None):
        self.trajectory_len = cfg.get('trajectory_len', 3) if cfg else 3
        self.use_meta = cfg.get('use_meta', False) if cfg else False
        if self.use_meta:
            self.count = 1
        self.step_data_compressor_name = cfg.get('step_data_compressor', 'none') if cfg else 'none'
        self.step_data_compressor = get_step_data_compressor(self.step_data_compressor_name)
        self.output_dir = './data'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def __len__(self):
        return 128  # pseudo length, only for implement interface

    def __getitem__(self, idx):
        if self.use_meta:
            # save only one fake data file to reduce resource waste
            path = os.path.join(self.output_dir, 'data_{}.pt'.format(1))
            if not os.path.exists(path):
                data = self.step_data_compressor(self.get_1v1_agent_data())
                torch.save(data, path)
            self.count += 1
            return {
                'job_id': self.count - 1,
                'trajectory_path': path,
                'priority': 1.0,
                'step_data_compressor': self.step_data_compressor_name,
            }
        else:
            return self.get_1v1_agent_data()

    def get_1v1_agent_data(self):
        def get_outputs(actions, entity_num):
            ret = {}
            ret['action_type'] = torch.rand(NUM_ACTION_TYPES)
            ret['delay'] = torch.rand(DELAY_MAX + 1)
            if isinstance(actions['queued'], type(NOOP)):
                ret['queued'] = NOOP
            else:
                ret['queued'] = torch.randn(2)
            if isinstance(actions['selected_units'], type(NOOP)):
                ret['selected_units'] = NOOP
            else:
                num = actions['selected_units'].shape[0]
                num = min(MAX_SELECTED_UNITS, num + 1)
                ret['selected_units'] = torch.rand(num, entity_num + 1)
            if isinstance(actions['target_units'], type(NOOP)):
                ret['target_units'] = NOOP
            else:
                ret['target_units'] = torch.rand(entity_num)
            if isinstance(actions['target_location'], type(NOOP)):
                ret['target_location'] = NOOP
            else:
                ret['target_location'] = torch.rand(*MAP_SIZE)
            return ret

        def disturb_outputs(outputs):
            new_outputs = copy.deepcopy(outputs)
            new_outputs['action_type'] += torch.randn_like(new_outputs['action_type']) * 0.1
            new_outputs['delay'] = torch.clamp(new_outputs['delay'] + torch.randn(1) * 10, 0, DELAY_MAX)
            if new_outputs['queued'] is not NOOP:
                new_outputs['queued'] += torch.randn_like(new_outputs['queued']) * 0.1
            if new_outputs['selected_units'] is not NOOP:
                new_outputs['selected_units'] += torch.randn_like(new_outputs['selected_units']) * 0.1
            if new_outputs['target_units'] is not NOOP:
                new_outputs['target_units'] += torch.randn_like(new_outputs['target_units']) * 0.1
            if new_outputs['target_location'] is not NOOP:
                new_outputs['target_location'] += torch.randn_like(new_outputs['target_location']) * 0.1
            return new_outputs

        def disturb_actions(actions):
            new_actions = copy.deepcopy(actions)
            if new_actions['selected_units'] is not NOOP:
                num = np.random.randint(-2, 2)
                if num > 0:
                    handle = new_actions['selected_units']
                    new_actions['selected_units'] = torch.cat([handle, handle[:num]], dim=0)
                elif num < 0:
                    new_actions['selected_units'] = new_actions['selected_units'][:num]
            return new_actions

        def get_single_rl_agent_step_data():
            base = get_single_step_data()
            base['score_cumulative'] = random_tensor([13])
            base['prev_state'] = [torch.zeros(*LSTM_DIMS), torch.zeros(*LSTM_DIMS)]
            base['rewards'] = get_fake_rewards()
            base['game_seconds'] = random.randint(0, 24 * 60)
            base['target_outputs'] = get_outputs(base['actions'], base['entity_info'].shape[0])
            base['behaviour_outputs'] = disturb_outputs(
                base['target_outputs']
            ) if np.random.random() > 0.3 else get_outputs(base['actions'], base['entity_info'].shape[0])
            if np.random.random() > 0.3:
                base['teacher_actions'] = copy.deepcopy(base['actions'])
                base['teacher_outputs'] = disturb_outputs(base['target_outputs'])
            else:
                base['teacher_actions'] = disturb_actions(base['actions'])
                base['teacher_outputs'] = get_outputs(base['teacher_actions'], base['entity_info'].shape[0])
            return base

        data = []
        for i in range(self.trajectory_len):
            data.append({'home': get_single_rl_agent_step_data(), 'away': get_single_rl_agent_step_data()})
        data[-1]['home_next'] = get_single_step_data()
        data[-1]['home_next']['score_cumulative'] = random_tensor([13])
        data[-1]['away_next'] = get_single_step_data()
        data[-1]['away_next']['score_cumulative'] = random_tensor([13])
        return data
