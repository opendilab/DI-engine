import os
import sys
import ceph
import torch
import numpy as np
import pickle

from sc2learner.utils import read_file_ceph
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from pysc2.lib.action_dict import ACTION_INFO_MASK, ACTIONS_STAT

result = {}


def save_obj(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def analyse_stat(stat_path):
    try:
        stat = read_file_ceph(stat_path.strip(), read_type='pickle')
        # stat = torch.load(stat)
        if 'action_statistics' in stat:
            for action_type, statistics in stat['action_statistics'].items():
                if action_type not in result:
                    result[action_type] = [set(), set()]
                if statistics['selected_type']:
                    result[action_type][0] = result[action_type][0] | statistics['selected_type']
                if statistics['target_type']:
                    result[action_type][1] = result[action_type][1] | statistics['target_type']
        return True
    except FileNotFoundError as e:
        print('failed to read {}'.format(stat_path))
        return False


def main():
    stat_list_path = '/mnt/lustre/zhangmanyuan/data_t1/nature-agi/Protoss.10000'
    ps = open(stat_list_path, 'r').readlines()

    for i, p in enumerate(ps):
        p = p.strip()
        p = p + '.stat'
        analyse_stat(p)
        # print(result)

    print('----------------------------------')
    # print(result)

    # save_obj(result, '/mnt/lustre/zhangming/data/stat_info.pkl')
    return result


def merge_units():
    ret = {}
    for x in [Neutral, Protoss, Terran, Zerg]:
        for item in x:
            assert item.value not in ret, '{} {} / {}'.format(item.value, item.name, ret[item.value])
            ret[item.value] = item.name
    return ret


if __name__ == '__main__':
    result = main()

    def merge_dict(d1, d2):
        key_names = ['action_name', 'selected_type', 'selected_type_name', 'target_type', 'target_type_name']
        assert d1['action_name'] == d2['action_name']
        temp_selected_1 = {k: v for k, v in zip(d1['selected_type'], d1['selected_type_name'])}
        temp_selected_2 = {k: v for k, v in zip(d2['selected_type'], d2['selected_type_name'])}
        for k, v in temp_selected_1.items():
            if k in temp_selected_2:
                assert temp_selected_1[k] == temp_selected_2[k]
        result_selected = {**temp_selected_1, **temp_selected_2}

        temp_target_1 = {k: v for k, v in zip(d1['target_type'], d1['target_type_name'])}
        temp_target_2 = {k: v for k, v in zip(d2['target_type'], d2['target_type_name'])}
        for k, v in temp_target_1.items():
            if k in temp_target_2:
                assert temp_target_1[k] == temp_target_2[k]
        result_target = {**temp_target_1, **temp_target_2}
        ret = {
            'action_name': d1['action_name'],
            'selected_type': list(result_selected.keys()),
            'selected_type_name': list(result_selected.values()),
            'target_type': list(result_target.keys()),
            'target_type_name': list(result_target.values())
        }
        return ret

    def print_dict(d):
        s = '{\n'
        for k, v in d.items():
            s += "    " + str(k) + ": {'action_name': '" + str(v['action_name']) + "'"
            s += ", 'selected_type': " + str(v['selected_type'])
            s += ", 'selected_type_name': " + str(v['selected_type_name'])
            s += ", 'target_type': " + str(v['target_type'])
            s += ", 'target_type_name': " + str(v['target_type_name'])
            s += "},\n"
        s += '}'
        return s

    result_dict = {}
    action_ids = list(result.keys())
    action_ids.sort()
    total_dict = merge_units()
    for action_id in action_ids:
        value = result[action_id]
        list_selected_type = list(value[0])
        list_target_type = list(value[1])
        action_name = ACTION_INFO_MASK[int(action_id)]['name']
        list_selected_type_name = [total_dict[int(selected_type)] for selected_type in list_selected_type]
        list_target_type_name = [total_dict[int(target_type)] for target_type in list_target_type]

        result_dict[action_id] = {
            'action_name': action_name,
            'selected_type': list_selected_type,
            'selected_type_name': list_selected_type_name,
            'target_type': list_target_type,
            'target_type_name': list_target_type_name
        }

    result_new = {}

    for k, v in ACTIONS_STAT.items():
        if k in result_dict:
            result_new[k] = merge_dict(ACTIONS_STAT[k], result_dict[k])
        else:
            result_new[k] = v

    for k, v in result_dict.items():
        if k not in result_new:
            result_new[k] = v

    with open('/mnt/lustre/zhangming/data/stat_info.txt', 'w') as f:
        s = print_dict(result_new)
        print(s)
        f.write(s)
