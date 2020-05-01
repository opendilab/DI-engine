import torch
import copy
import numpy as np
import logging
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
     UNIT_TYPES_REORDER_ARRAY, BUFFS_REORDER_ARRAY, ABILITIES_REORDER_ARRAY, NUM_UPGRADES, UPGRADES_REORDER,\
     UPGRADES_REORDER_ARRAY, NUM_ACTIONS, ACTIONS_REORDER_ARRAY, NUM_ADDON, ADDON_REORDER_ARRAY,\
     NUM_BEGIN_ACTIONS, NUM_UNIT_BUILD_ACTIONS, NUM_EFFECT_ACTIONS, NUM_RESEARCH_ACTIONS,\
     UNIT_BUILD_ACTIONS_REORDER_ARRAY, EFFECT_ACTIONS_REORDER_ARRAY, RESEARCH_ACTIONS_REORDER_ARRAY,\
     BEGIN_ACTIONS_REORDER_ARRAY, UNIT_BUILD_ACTIONS, EFFECT_ACTIONS, RESEARCH_ACTIONS, BEGIN_ACTIONS

# TODO: move these shared functions to utils
from sc2learner.envs.observations.alphastar_obs_wrapper import reorder_one_hot_array,\
     batch_binary_encode, div_one_hot, LOCATION_BIT_NUM


class Statistics:
    """
    Class carrying the game statistics of multiple players

    Args:
        player_num: number of players to be hold
        begin_num: how many of build actions need to be recorded in the statistics
            in L111 of detailed-architecture.txt, only the first 20 are stored
    """
    def __init__(self, player_num=2, begin_num=200):
        self.player_num = player_num
        self.action_statistics = [{} for _ in range(player_num)]
        self.cumulative_statistics = [{} for _ in range(player_num)]
        self.build_order_statistics = [[] for _ in range(player_num)]
        self.cached_transformed_stat = [None] * self.player_num
        self.cached_z = [None] * self.player_num
        self.begin_num = begin_num
        # according to detailed-arch L109, mmr is fixed to 6200 unless in supervised learning
        # this will not affect replay decoding, where the meta for transform_stat is externally supplied
        self.mmr = 6200

    def load_from_transformed_stat(self, transformed_stat, player, begin_num=None):
        """
        Loading cumulative_statistics and build_order_statistics
        produced by transform_stat/get_transformed_stat as a Statistics object
        as the count of actions is lost in the processing, the loaded action count will not be vaild

        Args:
            transformed_stat: input stat
            begin_num: the beginning_build_order will be cutted/padded with zeros to this length
            player: the index of player should the stat be loaded for

        Returns:
            None
        """
        transformed_stat = copy.deepcopy(transformed_stat)
        if begin_num is not None:
            transformed_stat['beginning_build_order'] = transformed_stat['beginning_build_order'][:begin_num]
            if transformed_stat['beginning_build_order'].shape[0] < begin_num:
                # filling zeros if there is too few begining_build_order entries
                B, N = transformed_stat['beginning_build_order'].shape
                B0 = begin_num - B
                transformed_stat['beginning_build_order'] = torch.cat(
                    [transformed_stat['beginning_build_order'],
                     torch.zeros(B0, N)]
                )
        # loading cumulative_stat
        bu = np.argwhere(transformed_stat['cumulative_stat']['unit_build'].numpy() == 1)
        for n in bu:
            act = UNIT_BUILD_ACTIONS[n[0]]
            self.update_cum_stat({'action_type': act}, player)
        eff = np.argwhere(transformed_stat['cumulative_stat']['effect'].numpy() == 1)
        for n in eff:
            act = EFFECT_ACTIONS[n[0]]
            self.update_cum_stat({'action_type': act}, player)
        rs = np.argwhere(transformed_stat['cumulative_stat']['research'].numpy() == 1)
        for n in rs:
            act = RESEARCH_ACTIONS[n[0]]
            self.update_cum_stat({'action_type': act}, player)
        # loading build_order_statistics
        bu_np = transformed_stat['beginning_build_order'].numpy()
        bu = np.argwhere(bu_np[:, :-2 * LOCATION_BIT_NUM])
        weight_arr = 2**np.arange(LOCATION_BIT_NUM - 1, -1, -1)
        for n in bu:
            x = np.sum(bu_np[n[0], -2 * LOCATION_BIT_NUM:-1 * LOCATION_BIT_NUM] * weight_arr)
            y = np.sum(bu_np[n[0], -1 * LOCATION_BIT_NUM:] * weight_arr)
            self.build_order_statistics[player].append({'action_type': BEGIN_ACTIONS[n[1]], 'location': [x, y]})
        transformed_stat['mmr'] = div_one_hot(torch.LongTensor([self.mmr]), 6000, 1000).squeeze(0)
        self.cached_transformed_stat[player] = transformed_stat
        self.cached_z = [None] * self.player_num

    def update_action_stat(self, act, obs, player):
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
        if action_type not in self.action_statistics[player].keys():
            self.action_statistics[player][action_type] = {
                'count': 0,
                'selected_type': set(),
                'target_type': set(),
            }
        self.action_statistics[player][action_type]['count'] += 1
        entity_type_dict = {id: type for id, type in zip(obs['entity_raw']['id'], obs['entity_raw']['type'])}
        if isinstance(act['selected_units'], torch.Tensor):
            units = act['selected_units'].tolist()
            unit_types = get_unit_types(units, entity_type_dict)
            self.action_statistics[player][action_type]['selected_type'] =\
                self.action_statistics[player][action_type]['selected_type'].union(
                unit_types
            )  # noqa
        if isinstance(act['target_units'], torch.Tensor):
            units = act['target_units'].tolist()
            unit_types = get_unit_types(units, entity_type_dict)
            self.action_statistics[player][action_type]['target_type'] = self.action_statistics[player][action_type][
                'target_type'].union(unit_types)  # noqa

    def update_cum_stat(self, act, player):
        # this will not clear the cache
        action_type = int(act['action_type'])
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if goal != 'other':
            if action_type not in self.cumulative_statistics[player].keys():
                self.cumulative_statistics[player][action_type] = {'count': 1, 'goal': goal}
            else:
                self.cumulative_statistics[player][action_type]['count'] += 1

    def update_build_order_stat(self, act, player):
        # this will not clear the cache
        target_list = ['unit', 'build', 'research']
        action_type = int(act['action_type'])
        if action_type in (35, 64, 520, 222, 515, 503):  # exclude worker and supply
            return
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if goal in target_list:
            if goal == 'build':
                if act['target_location'] is None:
                    print(
                        'build action have no target_location!'
                        'this shouldn\'t happen with real model: {}'.format(act)
                    )
                location = act['target_location']
                if isinstance(location, torch.Tensor):  # for build ves, no target_location
                    location = location.tolist()
            else:
                location = 'none'
            self.build_order_statistics[player].append({'action_type': action_type, 'location': location})

    def update_stat(self, act, obs, player):
        """
        Update action_stat cum_stat and build_order_stat

        Args:
            act: Processed general action
            obs: observation
            player: index of the player to update
        """
        self.cached_transformed_stat[player] = None
        self.cached_z[player] = None
        self.update_action_stat(act, obs, player)
        self.update_cum_stat(act, player)
        if len(self.build_order_statistics[player]) < self.begin_num:
            self.update_build_order_stat(act, player)

    def get_stat(self, player=None):
        """
        Get raw statistics data (before transformation to tensor input)

        Args:
            player: index of player or None for getting a list of all players

        Return:
            stat: a dict with keys of 'action_statistics', 'cumulative_statistics', 'begin_statistics'
                or a list of dicts if player=None
        """
        if player is None:
            return [
                {
                    'action_statistics': self.action_statistics[idx],
                    'cumulative_statistics': self.cumulative_statistics[idx],
                    'begin_statistics': self.build_order_statistics[idx]
                } for idx in range(self.player_num)
            ]
        else:
            return {
                'action_statistics': self.action_statistics[player],
                'cumulative_statistics': self.cumulative_statistics[player],
                'begin_statistics': self.build_order_statistics[player]
            }

    def get_transformed_cum_stat(self, player):
        return transform_cum_stat(self.cumulative_statistics[player])

    def get_transformed_stat(self, player=None, mmr=None):
        '''
        Export the statistics as transformed stat, which is ready as the network input
        Args:
            player: the index of the player or None if requesting for all players as a list
            mmr: the mmr to be encoded in the tensors, if set to None, self.mmr=6200 will be used
        Returns:
            stat: encoded stat dict like
            {
                'mmr': mmr,  # one hot encoded mmr
                'beginning_build_order': beginning_build_order_tensor,
                'cumulative_stat': cumulative_stat_tensor
            }
            all tensors are LongTensor
            this is also the format accepted by load_from_transformed_stat
        '''
        if mmr is None:
            mmr = self.mmr
        if player is None:
            ret = []
            for player in range(self.player_num):
                if self.cached_transformed_stat[player] is not None:
                    ret.append(self.cached_transformed_stat[player])
                else:
                    meta = {'home_mmr': mmr}
                    tstat = transform_stat(self.get_stat(player), meta)
                    ret.append(tstat)
                    self.cached_transformed_stat[player] = copy.deepcopy(tstat)
            return ret
        else:
            if self.cached_transformed_stat[player] is not None:
                return self.cached_transformed_stat[player]
            else:
                meta = {'home_mmr': mmr}
                tstat = transform_stat(self.get_stat(player), meta)
                self.cached_transformed_stat[player] = copy.deepcopy(tstat)
                return tstat

    def get_z(self, idx):
        '''
        Export the cum and build statistics
        in an alternative format used for computing RL training baselines
        Args:
            idx: the index of the player
        Returns:
            a dict with keys 'built_units', 'effects', 'upgrades', 'build_order'
            note: the actions in build_order is raw_ability
        '''
        if self.cached_z[idx] is not None:
            return self.cached_z[idx]
        cum_stat_tensor = transform_cum_stat(self.cumulative_statistics[idx])
        ret = {
            'built_units': cum_stat_tensor['unit_build'],
            'effects': cum_stat_tensor['effect'],
            'upgrades': cum_stat_tensor['research'],
            'build_order': transform_build_order_to_z_format(self.build_order_statistics[idx])
        }
        self.cached_z[idx] = copy.deepcopy(ret)
        return ret


def transform_build_order_to_z_format(stat):
    '''Used internally by Statistics.get_z()
    '''
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


def transform_cum_stat(cumulative_stat):
    cumulative_stat_tensor = {
        'unit_build': torch.zeros(NUM_UNIT_BUILD_ACTIONS),
        'effect': torch.zeros(NUM_EFFECT_ACTIONS),
        'research': torch.zeros(NUM_RESEARCH_ACTIONS)
    }
    for k, v in cumulative_stat.items():
        if v['goal'] in ['unit', 'build']:
            cumulative_stat_tensor['unit_build'][UNIT_BUILD_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['effect']:
            cumulative_stat_tensor['effect'][EFFECT_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['research']:
            cumulative_stat_tensor['research'][RESEARCH_ACTIONS_REORDER_ARRAY[k]] = 1
    return cumulative_stat_tensor


def transform_stat(stat, meta, location_num=LOCATION_BIT_NUM):
    beginning_build_order = stat['begin_statistics']
    beginning_build_order_tensor = []
    for item in beginning_build_order:
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
    cumulative_stat_tensor = transform_cum_stat(stat['cumulative_statistics'])
    mmr = meta['home_mmr']
    mmr = torch.LongTensor([mmr])
    mmr = div_one_hot(mmr, 6000, 1000).squeeze(0)
    return {
        'mmr': mmr,
        'beginning_build_order': beginning_build_order_tensor,
        'cumulative_stat': cumulative_stat_tensor
    }  # noqa
