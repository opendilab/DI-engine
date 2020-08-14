'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. filter replays with specific constraints and generate
    the filterd replay list
'''
import os
import random
import torch
from nervex.utils import read_file_ceph


def generate_sl_data_list(
    replay_list,
    output_dir,
    min_mmr=0,
    home_race=None,
    away_race=None,
    trajectory_len=64,
    target_map=None,
    train_ratio=0.95
):
    race_list = ['Protoss', 'Terran', 'Zerg']
    map_list = [
        'Kairos Junction LE', 'New Repugnancy LE', 'Cyber Forest LE', "King's Cove LE", "Turbo Cruise '84 LE",
        'Thunderbird LE', 'Acropolis LE'
    ]
    assert (home_race is None or home_race in race_list)
    assert (away_race is None or away_race in race_list)
    assert (target_map is None or target_map in map_list)
    with open(replay_list, 'r') as f:
        data = f.readlines()

    replay_record_dict = {}
    valid_replay = []
    for idx, item in enumerate(data):
        name, ext = item[:-1].split('.')
        if name not in replay_record_dict.keys():
            replay_record_dict[name] = set([ext])
        else:
            replay_record_dict[name].add(ext)
            if len(replay_record_dict[name]) == 4:
                valid_replay.append(name)
    print("===================valid parse finish=======================")

    selected_replay = []
    map_set = set()
    for idx, item in enumerate(valid_replay):
        meta_item = item + '.meta'
        try:
            meta_item = read_file_ceph(meta_item)
        except Exception as e:
            print("read file error: {}".format(e))
            continue
        meta = torch.load(meta_item)
        map_set.add(meta['map_name'])
        if meta['home_mmr'] < min_mmr or meta['away_mmr'] < min_mmr:
            continue
        if home_race and meta['home_race'] != home_race:
            continue
        if away_race and meta['away_race'] != away_race:
            continue
        if target_map and meta['map_name'] != target_map:
            continue
        if meta['step_num'] < trajectory_len:
            continue
        selected_replay.append(item)
    print("===================select finish=======================")
    print('map_set', map_set)

    random.shuffle(selected_replay)
    num = int(len(selected_replay) * train_ratio)
    train = selected_replay[:num]
    train = [t + '\n' for t in train]
    val = selected_replay[num:]
    val = [t + '\n' for t in val]

    def remove_space(s):
        return s.replace(" ", "") if s is not None else s

    prefix = '{}_{}_{}_{}'.format(home_race, away_race, remove_space(target_map), min_mmr)
    output_name = prefix + '_train_{}.txt'.format(len(train))

    with open(output_name, 'w') as f:
        f.writelines(train)

    output_name = prefix + '_val_{}.txt'.format(len(val))

    with open(output_name, 'w') as f:
        f.writelines(val)


'''
Note:
    if you want to generate mini SL data list, set:
        - home_race = 'Zerg'
        - away_race = 'Zerg'
        - target_map = 'Kairos Junction LE'
'''
home_race = 'Zerg'
away_race = None
target_map = None
trajectory_len = 64
min_mmr = 3500
replay_list = '/mnt/lustre/zhangming/data/replay_decode_410_clean.train.list'
output_dir = '.'

if __name__ == "__main__":
    generate_sl_data_list(replay_list, output_dir, min_mmr, home_race, away_race, trajectory_len, target_map)
