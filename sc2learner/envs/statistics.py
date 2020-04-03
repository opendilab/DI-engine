import torch
import logging
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
     UNIT_TYPES_REORDER_ARRAY, BUFFS_REORDER_ARRAY, ABILITIES_REORDER_ARRAY, NUM_UPGRADES, UPGRADES_REORDER,\
     UPGRADES_REORDER_ARRAY, NUM_ACTIONS, ACTIONS_REORDER_ARRAY, NUM_ADDON, ADDON_REORDER_ARRAY,\
     NUM_BEGIN_ACTIONS, NUM_UNIT_BUILD_ACTIONS, NUM_EFFECT_ACTIONS, NUM_RESEARCH_ACTIONS,\
     UNIT_BUILD_ACTIONS_REORDER_ARRAY, EFFECT_ACTIONS_REORDER_ARRAY, RESEARCH_ACTIONS_REORDER_ARRAY,\
     BEGIN_ACTIONS_REORDER_ARRAY

# TODO: move these shared functions to utils
from sc2learner.envs.observations.alphastar_obs_wrapper import reorder_one_hot_array,\
     batch_binary_encode, div_one_hot, LOCATION_BIT_NUM


class Statistics:
    def __init__(self, player_num=2, begin_num=200):
        self.player_num = player_num
        self.action_statistics = [{} for _ in range(player_num)]
        self.cumulative_statistics = [{} for _ in range(player_num)]
        self.begin_statistics = [[] for _ in range(player_num)]
        self.begin_num = begin_num

    def update_action_stat(self, act, obs, player):
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
            self.action_statistics[player][action_type]['target_type'] = self.action_statistics[player][action_type]['target_type'].union(
                unit_types
            )  # noqa

    def update_cum_stat(self, act, player):
        action_type = int(act['action_type'])
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if goal != 'other':
            if action_type not in self.cumulative_statistics[player].keys():
                self.cumulative_statistics[player][action_type] = {'count': 1, 'goal': goal}
            else:
                self.cumulative_statistics[player][action_type]['count'] += 1

    def update_begin_stat(self, act, player):
        target_list = ['unit', 'build', 'research', 'effect']
        action_type = int(act['action_type'])
        goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
        if goal in target_list:
            if goal == 'build':
                location = act['target_location']
                if isinstance(location, torch.Tensor):  # for build ves, no target_location
                    location = location.tolist()
            else:
                location = 'none'
            self.begin_statistics[player].append({'action_type': action_type, 'location': location})

    def update_stat(self, act, obs, player):
        """
        Update action_stat cum_stat and begin_stat
        act should be preprocessed general action
        """
        self.update_action_stat(act, obs, player)
        self.update_cum_stat(act, player)
        if len(self.begin_statistics[player]) < self.begin_num:
            self.update_begin_stat(act, player)

    def get_transformed_cum_stat(self, player):
        return transform_cum_stat(self.cumulative_statistics[player])

    def get_stat(self):
        return [
                {
                    'action_statistics': self.action_statistics[idx],
                    'cumulative_statistics': self.cumulative_statistics[idx],
                    'begin_statistics': self.begin_statistics[idx]
                } for idx in range(self.player_num)
            ]

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

