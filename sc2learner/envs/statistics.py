import os
import pickle
from collections import namedtuple
import copy
import numpy as np
import torch
import logging
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
     UNIT_TYPES_REORDER_ARRAY, BUFFS_REORDER_ARRAY, ABILITIES_REORDER_ARRAY, NUM_UPGRADES, UPGRADES_REORDER,\
     UPGRADES_REORDER_ARRAY, NUM_ACTIONS, ACTIONS_REORDER_ARRAY, NUM_ADDON, ADDON_REORDER_ARRAY,\
     NUM_BEGIN_ACTIONS, NUM_UNIT_BUILD_ACTIONS, NUM_EFFECT_ACTIONS, NUM_RESEARCH_ACTIONS,\
     UNIT_BUILD_ACTIONS_REORDER_ARRAY, EFFECT_ACTIONS_REORDER_ARRAY, RESEARCH_ACTIONS_REORDER_ARRAY,\
     BEGIN_ACTIONS_REORDER_ARRAY, UNIT_BUILD_ACTIONS, EFFECT_ACTIONS, RESEARCH_ACTIONS, BEGIN_ACTIONS,\
     OLD_BEGIN_ACTIONS_REORDER_INV

# TODO: move these shared functions to utils
from sc2learner.envs.observations.alphastar_obs_wrapper import reorder_one_hot_array,\
     batch_binary_encode, div_one_hot, LOCATION_BIT_NUM
from sc2learner.torch_utils import to_dtype, one_hot


def binary_search(data, item):
    if len(data) <= 0:
        raise RuntimeError("empty data with len: {}".format(len(data)))
    low = 0
    high = len(data) - 1
    while low <= high:
        mid = (high + low) // 2
        if data[mid] == item:
            return mid
        elif data[mid] < item:
            low = mid + 1
        else:
            high = mid - 1
    if low == len(data):
        low -= 1  # limit low within [0, len(data)-1]
    return low


class RealTimeStatistics:
    """
    Overview: real time agent statistics
    """
    def __init__(self, begin_num=20):
        self.action_statistics = {}
        self.cumulative_statistics = {}
        self.cumulative_statistics_game_loop = []
        self.begin_statistics = []
        self.begin_num = begin_num

    def update_action_stat(self, act, obs):
        # this will not clear the cache

        def get_unit_types(units, entity_type_dict):
            unit_types = set()
            for u in units:
                try:
                    unit_type = entity_type_dict[u]
                    unit_types.add(unit_type)
                except KeyError:
                    logging.warning("Not found unit(id: {})".format(u))
            return unit_types

        action_type = int(act['action_type'])  # this can accept either torch.LongTensor and int
        if action_type not in self.action_statistics.keys():
            self.action_statistics[action_type] = {
                'count': 0,
                'selected_type': set(),
                'target_type': set(),
            }
        self.action_statistics[action_type]['count'] += 1
        entity_type_dict = {id: type for id, type in zip(obs['entity_raw']['id'], obs['entity_raw']['type'])}
        if isinstance(act['selected_units'], torch.Tensor):
            units = act['selected_units'].tolist()
            unit_types = get_unit_types(units, entity_type_dict)
            self.action_statistics[action_type]['selected_type'] =\
                self.action_statistics[action_type]['selected_type'].union(
                unit_types
            )  # noqa
        if isinstance(act['target_units'], torch.Tensor):
            units = act['target_units'].tolist()
            unit_types = get_unit_types(units, entity_type_dict)
            self.action_statistics[action_type]['target_type'] = self.action_statistics[action_type][
                'target_type'].union(unit_types)  # noqa

    def update_cum_stat(self, act, game_loop):
        # this will not clear the cache
        action_type = int(act['action_type'])
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if goal != 'other':
            if action_type not in self.cumulative_statistics.keys():
                self.cumulative_statistics[action_type] = {'count': 1, 'goal': goal}
            else:
                self.cumulative_statistics[action_type]['count'] += 1
            loop_stat = copy.deepcopy(self.cumulative_statistics)
            loop_stat['game_loop'] = game_loop
            self.cumulative_statistics_game_loop.append(loop_stat)

    def update_build_order_stat(self, act, game_loop):
        # this will not clear the cache
        action_type = int(act['action_type'])
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if action_type in BEGIN_ACTIONS:
            if goal == 'build':
                if action_type not in [36, 197, 214] and act['target_location'] is None:
                    print(
                        'build action have no target_location!'
                        'this shouldn\'t happen with real model: {}'.format(act)
                    )
                location = act['target_location']
                if isinstance(location, torch.Tensor):  # for build ves, no target_location
                    location = location.tolist()
            else:
                location = 'none'
            self.begin_statistics.append({'action_type': action_type, 'location': location, 'game_loop': game_loop})

    def update_stat(self, act, obs, game_loop):
        """
        Update action_stat cum_stat and build_order_stat

        Args:
            act: Processed general action
            obs: observation
            game_loop: current game loop
        """
        if obs is not None:
            self.update_action_stat(act, obs)
        self.update_cum_stat(act, game_loop)
        self.update_build_order_stat(act, game_loop)

    def get_reward_z(self, use_max_bo_clip):
        beginning_build_order = self.begin_statistics
        if use_max_bo_clip and len(beginning_build_order) > self.begin_num:
            beginning_build_order = beginning_build_order[:self.begin_num]
        cumulative_stat = self.cumulative_statistics
        cum_stat_tensor = transform_cum_stat(cumulative_stat)
        ret = {
            'built_unit': cum_stat_tensor['unit_build'],
            'effect': cum_stat_tensor['effect'],
            'upgrade': cum_stat_tensor['research'],
            'build_order': transform_build_order_to_z_format(beginning_build_order),
        }
        ret = to_dtype(ret, torch.long)
        return ret


class GameLoopStatistics:
    """
    Overview: Human replay data statistics specified by game loop
    """
    def __init__(self, stat, begin_num=20):
        self.ori_stat = stat
        self.ori_stat = self.add_game_loop(self.ori_stat)
        self.begin_num = begin_num
        self.mmr = 6200
        self._clip_global_bo()
        self.cache_reward_z = None
        self.cache_input_z = None
        self.max_game_loop = self.ori_stat['cumulative_stat'][-1]['game_loop']
        self._init_global_z()

    def add_game_loop(self, stat):
        beginning_build_order = stat['beginning_build_order']
        cumulative_stat = stat['cumulative_stat']
        if 'game_loop' in beginning_build_order[0].keys():
            return stat

        def is_action_frame(action_type, cum_idx):
            last_frame = cumulative_stat[cum_idx - 1]
            cur_frame = cumulative_stat[cum_idx]
            miss_key = cur_frame.keys() - last_frame.keys()
            diff_count_key = set()
            for k in last_frame.keys():
                if k != 'game_loop' and cur_frame[k]['count'] != last_frame[k]['count']:
                    diff_count_key.add(k)
            diff_key = miss_key.union(diff_count_key)
            return action_type in diff_key

        cum_idx = 1
        new_beginning_build_order = []
        for i in range(len(beginning_build_order)):
            item = beginning_build_order[i]
            action_type = item['action_type']
            while cum_idx < len(cumulative_stat) and not is_action_frame(action_type, cum_idx):
                cum_idx += 1
            if cum_idx < len(cumulative_stat):
                item.update({'game_loop': cumulative_stat[cum_idx]['game_loop']})
                new_beginning_build_order.append(item)

        new_stat = stat
        new_stat['beginning_build_order'] = new_beginning_build_order
        new_stat['begin_game_loop'] = [t['game_loop'] for t in new_beginning_build_order]
        new_stat['cum_game_loop'] = [t['game_loop'] for t in new_stat['cumulative_stat']]
        return new_stat

    def _clip_global_bo(self):
        beginning_build_order = copy.deepcopy(self.ori_stat['beginning_build_order'])
        if len(beginning_build_order) < self.begin_num:
            miss_num = self.begin_num - len(beginning_build_order)
            pad_beginning_build_order = beginning_build_order + [
                {
                    'action_type': 0,
                    'location': 'none'
                } for _ in range(miss_num)
            ]
            self.input_global_bo = pad_beginning_build_order
            self.reward_global_bo = beginning_build_order
        else:
            beginning_build_order = beginning_build_order[:self.begin_num]
            self.input_global_bo = beginning_build_order
            self.reward_global_bo = beginning_build_order

    def _init_global_z(self):
        # init input_global_z
        beginning_build_order, cumulative_stat = self.input_global_bo, self.ori_stat['cumulative_stat'][-1]
        self.input_global_z = transformed_stat_mmr(
            {
                'begin_statistics': beginning_build_order,
                'cumulative_statistics': cumulative_stat
            }, self.mmr
        )
        # init reward_global_z
        beginning_build_order, cumulative_stat = self.reward_global_bo, self.ori_stat['cumulative_stat'][-1]
        cum_stat_tensor = transform_cum_stat(cumulative_stat)
        self.reward_global_z = {
            'built_unit': cum_stat_tensor['unit_build'],
            'effect': cum_stat_tensor['effect'],
            'upgrade': cum_stat_tensor['research'],
            'build_order': transform_build_order_to_z_format(beginning_build_order),
        }
        self.reward_global_z = to_dtype(self.reward_global_z, torch.long)

    def get_input_z_by_game_loop(self, game_loop, cumulative_stat=None):
        """
        Note: if game_loop is None, load global stat
        """
        if cumulative_stat is None:
            if game_loop is None:
                return self.input_global_z
            else:
                _, cumulative_stat = self._get_stat_by_game_loop(game_loop)
        beginning_build_order = self.input_global_bo
        ret = transformed_stat_mmr(
            {
                'begin_statistics': beginning_build_order,
                'cumulative_statistics': cumulative_stat
            }, self.mmr
        )
        return ret

    def get_reward_z_by_game_loop(self, game_loop, build_order_length=None):
        """
        Note: if game_loop is None, load global stat
        """
        if game_loop is None:
            global_z = copy.deepcopy(self.reward_global_z)
            global_z['build_order']['type'] = global_z['build_order']['type'][:build_order_length]
            global_z['build_order']['loc'] = global_z['build_order']['loc'][:build_order_length]
            return global_z
        else:
            beginning_build_order, cumulative_stat = self._get_stat_by_game_loop(game_loop)
        # TODO(nyz) same game_loop case
        cum_stat_tensor = transform_cum_stat(cumulative_stat)
        ret = {
            'built_unit': cum_stat_tensor['unit_build'],
            'effect': cum_stat_tensor['effect'],
            'upgrade': cum_stat_tensor['research'],
            'build_order': transform_build_order_to_z_format(beginning_build_order),
        }
        ret = to_dtype(ret, torch.long)
        return ret

    def _get_stat_by_game_loop(self, game_loop):
        begin_idx = binary_search(self.ori_stat['begin_game_loop'], game_loop)
        cum_idx = binary_search(self.ori_stat['cum_game_loop'], game_loop)
        return self.ori_stat['beginning_build_order'][:begin_idx + 1], self.ori_stat['cumulative_stat'][cum_idx]

    def excess_max_game_loop(self, agent_game_loop):
        return agent_game_loop > self.max_game_loop


def transform_build_order_to_z_format(stat):
    """
    Overview: transform beginning_build_order to the format to calculate reward
    stat: list->element: dict('action_type': int, 'location': list(len=2)->element: int)
    """
    ret = {'type': np.zeros(len(stat), dtype=np.int), 'loc': np.empty((len(stat), 2), dtype=np.int)}
    zeroxy = np.array([0, 0], dtype=np.int)
    for n in range(len(stat)):
        action_type, location = stat[n]['action_type'], stat[n]['location']
        ret['type'][n] = action_type
        # TODO: check this 'loc' is x,y or y,x
        ret['loc'][n] = location if location is not None and location != 'none' else zeroxy
    ret['type'] = torch.Tensor(ret['type'])
    ret['loc'] = torch.Tensor(ret['loc'])
    return ret


def transform_build_order_to_input_format(stat, location_num=LOCATION_BIT_NUM):
    """
    Overview: transform beginning_build_order to the format for input
    stat: list->element: dict('action_type': int, 'location': list(len=2)->element: int)
    """
    beginning_build_order_tensor = []
    for item in stat:
        action_type, location = item['action_type'], item['location']
        action_type = torch.LongTensor([action_type])
        action_type = reorder_one_hot_array(action_type, BEGIN_ACTIONS_REORDER_ARRAY, num=NUM_BEGIN_ACTIONS)
        if location == 'none':
            location = torch.zeros(location_num * 2)
        else:
            x = batch_binary_encode(torch.LongTensor([location[0]]), bit_num=location_num)[0]
            y = batch_binary_encode(torch.LongTensor([location[1]]), bit_num=location_num)[0]
            location = torch.cat([x, y], dim=0)
        beginning_build_order_tensor.append(torch.cat([action_type.squeeze(0), location], dim=0))
    beginning_build_order_tensor = torch.stack(beginning_build_order_tensor, dim=0)
    return beginning_build_order_tensor


def transform_cum_stat(cumulative_stat):
    """
    Overview: transform cumulative_stat to the format for both input and reward
    cumulative_stat: dict('action_type': {'goal': str, count: int})
    """
    cumulative_stat_tensor = {
        'unit_build': torch.zeros(NUM_UNIT_BUILD_ACTIONS),
        'effect': torch.zeros(NUM_EFFECT_ACTIONS),
        'research': torch.zeros(NUM_RESEARCH_ACTIONS)
    }
    for k, v in cumulative_stat.items():
        if k == 'game_loop':
            continue
        if v['goal'] in ['unit', 'build']:
            cumulative_stat_tensor['unit_build'][UNIT_BUILD_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['effect']:
            cumulative_stat_tensor['effect'][EFFECT_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['research']:
            cumulative_stat_tensor['research'][RESEARCH_ACTIONS_REORDER_ARRAY[k]] = 1
    return cumulative_stat_tensor


def transform_stat(stat, meta, location_num=LOCATION_BIT_NUM):
    mmr = meta['home_mmr']
    return transformed_stat_mmr(stat, mmr, location_num)


def transformed_stat_mmr(stat, mmr, location_num=LOCATION_BIT_NUM):
    """
    Overview: transform replay metadata and statdata to input stat(mmr + z)
    """
    beginning_build_order = stat['begin_statistics']
    beginning_build_order_tensor = transform_build_order_to_input_format(beginning_build_order)
    cumulative_stat_tensor = transform_cum_stat(stat['cumulative_statistics'])
    mmr = torch.LongTensor([mmr])
    mmr = div_one_hot(mmr, 6000, 1000).squeeze(0)
    return {
        'mmr': mmr,
        'beginning_build_order': beginning_build_order_tensor,
        'cumulative_stat': cumulative_stat_tensor
    }


def transform_stat_processed(old_stat_processed):
    """
    Overview: transform new begin action(for stat_processed)
    """
    new_stat_processed = copy.deepcopy(old_stat_processed)
    beginning_build_order = new_stat_processed['beginning_build_order']
    new_beginning_build_order = []
    location_dim = 2 * LOCATION_BIT_NUM
    for item in beginning_build_order:
        action_type, location = item[:-location_dim], item[-location_dim:]
        action_type = torch.nonzero(action_type).item()
        action_type = OLD_BEGIN_ACTIONS_REORDER_INV[action_type]
        if action_type not in BEGIN_ACTIONS:
            continue
        action_type = BEGIN_ACTIONS_REORDER_ARRAY[action_type]
        action_type = torch.LongTensor([action_type])
        action_type = one_hot(action_type, NUM_BEGIN_ACTIONS)[0]
        new_item = torch.cat([action_type, location], dim=0)
        new_beginning_build_order.append(new_item)
    new_stat_processed['beginning_build_order'] = torch.stack(new_beginning_build_order, dim=0)
    return new_stat_processed


def transform_stat_professional_player(old_stat):
    new_stat = copy.deepcopy(old_stat)
    beginning_build_order = new_stat['beginning_build_order']
    new_beginning_build_order = []
    for item in beginning_build_order:
        if item['action_type'] in BEGIN_ACTIONS:
            new_beginning_build_order.append(item)
    new_stat['beginning_build_order'] = new_beginning_build_order
    return new_stat


class StatKey:
    def __init__(self, home_race=None, away_race=None, map_name=None, player_id=None):
        self.home_race = home_race
        self.away_race = away_race
        self.map_name = map_name
        self.player_id = player_id

    @classmethod
    def check_path(cls, item):
        """
        Overview: check stat path name format
        Note:
            format template: homerace_awayrace_mapname_playerid_id
        """
        race_list = ['zerg', 'terran', 'protoss']
        map_list = ['KingsCove', 'KairosJunction', 'NewRepugnancy', 'CyberForest']
        try:
            item_contents = item.split('_')
            assert len(item_contents) == 5
            assert item_contents[0] in race_list
            assert item_contents[1] in race_list
            assert item_contents[2] in map_list
            assert item_contents[3] in ['1', '2']
        except Exception as e:
            print(item_contents)
            return False
        return True

    @classmethod
    def path2key(cls, path):
        items = path.split('_')[:4]
        return StatKey(*items)

    def match(self, other):
        assert isinstance(other, StatKey)
        for k, v in self.__dict__.items():
            if v is not None:
                if other.__dict__[k] != v:
                    return False
        return True


class StatManager:
    def __init__(self, stat_dir):
        assert os.path.exists(stat_dir), stat_dir
        self.stat_dir = stat_dir
        self.stat_paths = [item for item in os.listdir(self.stat_dir) if StatKey.check_path(item)]
        self.stat_keys = [StatKey.path2key(t) for t in self.stat_paths]

    def get_ava_stats(self, **kwargs):
        assert kwargs['player_id'] == 'ava'
        # select matched results
        stats = []
        for player_id in ['1', '2']:
            kwargs['player_id'] = player_id
            query = StatKey(**kwargs)
            matched_results_idx = [idx for idx, t in enumerate(self.stat_keys) if query.match(t)]
            if len(matched_results_idx) == 0:
                raise RuntimeError("no matched stat, input kwargs are: {}".format(kwargs))
            # random sample
            selected_idx = np.random.choice(matched_results_idx)
            stat_path = self.stat_paths[selected_idx]
            stats.append(self._load_stat(stat_path))
        return stats

    def _load_stat(self, path):
        with open(os.path.join(self.stat_dir, path), 'rb') as f:
            stat = pickle.load(f)
        return stat
