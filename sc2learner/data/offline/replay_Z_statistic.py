# coding: utf-8
'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Generate human strategy which is Z in Alphastar from replays

Note this runs on Windows and requires various game versions
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import sys
import copy
import queue
import random
from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import torch
from pysc2 import run_configs
from pysc2.lib import replay
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser

PLAYER_NUM = 2
PROCESS_NUM = 2
FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "E:/data/replays", "Path to a directory of replays.")
flags.DEFINE_string("output_dir", "E:/data/replay_data", "Path to save data")
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("output_dir")
flags.FLAGS(sys.argv)


class ReplayDecoder(multiprocessing.Process):
    def __init__(self, replay_queue, serial_queue, output_dir, success_msg, error_msg):
        super(ReplayDecoder, self).__init__()
        self.run_config = run_configs.get()
        self.replay_queue = replay_queue
        self.serial_queue = serial_queue
        self.output_dir = output_dir
        self.success_msg = success_msg
        self.error_msg = error_msg
        self.interface = sc_pb.InterfaceOptions(
            raw=True, score=False, raw_crop_to_playable_area=True, feature_layer=sc_pb.SpatialCameraSetup(width=24)
        )

    def replay_decode(self, controller, replay_path, player, game_loops):
        map_size = None

        def update_cum_stat(cumulative_statistics, act, game_loop, cumulative_z):
            action_type = act['action_type'].item()
            goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
            if goal != 'other':
                if action_type not in cumulative_statistics.keys():
                    cumulative_statistics[action_type] = {'count': 1, 'goal': goal}
                else:
                    cumulative_statistics[action_type]['count'] += 1
                cumulative_statistics['game_loop'] = game_loop
                loop_stat = copy.deepcopy(cumulative_statistics)
                cumulative_z.append(loop_stat)

        def update_begin_stat(begin_statistics, act):
            target_list = ['unit', 'build', 'research', 'effect']
            action_type = act['action_type'].item()
            goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
            if goal in target_list:
                if goal == 'build':
                    location = act['target_location']
                    if isinstance(location, torch.Tensor):  # for build ves, no target_location
                        location = location.tolist()
                else:
                    location = 'none'
                begin_statistics.append({'action_type': action_type, 'location': location})

        # get actions first
        cumulative_statistics = {}
        cumulative_z = []
        begin_statistics = []
        born_location = [[] for _ in range(PLAYER_NUM)]
        begin_num = 200
        controller.start_replay(
            sc_pb.RequestStartReplay(
                replay_path=replay_path,
                options=self.interface,
                observed_player_id=player + 1,
            )
        )
        game_info = controller.game_info()
        map_size = game_info.start_raw.map_size
        ob = controller.observe()
        # get self born location
        location = []
        for i in ob.observation.raw_data.units:
            if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
                location.append([i.pos.x, i.pos.y])
        assert len(location) == 1, 'no fog'
        born_location[0] = location[0]
        # get opponent born location
        start_locations = []
        for i in game_info.start_raw.start_locations:
            start_locations.append(i)
        assert len(location) == 1, 'start location wrong'
        born_location[1] = [start_locations[0].x, start_locations[0].y]
        act_parser = AlphastarActParser(feature_layer_resolution=1, map_size=[map_size.x, map_size.y])
        cur_loop = 0
        while cur_loop < game_loops:
            next_loop = min(game_loops, cur_loop + 1000)
            try:
                controller.step(next_loop - cur_loop)
                ob = controller.observe()
                for i in ob.actions:
                    if i.HasField('action_raw'):
                        assert i.HasField('game_loop'), 'action no game_loop'  # debug
                        agent_act = act_parser.parse(i.action_raw)
                        agent_act = act_parser.merge_same_id_action(agent_act)[0]
                        update_cum_stat(cumulative_statistics, agent_act, i.game_loop, cumulative_z)
                        if len(begin_statistics) < begin_num:
                            update_begin_stat(begin_statistics, agent_act)
            except:
                if next_loop == game_loops:
                    pass
                else:
                    raise Exception('decoding went wrong')
            cur_loop = next_loop

        return ({'cumulative_statistics': cumulative_z, 'begin_statistics': begin_statistics}, map_size, born_location)

    def parse_info(self, info, replay_path):
        if (info.player_info[0].player_info.race_actual != 2 and info.player_info[1].player_info.race_actual != 2):
            # not include Zerg race
            return None
        race_dict = {1: 'Terran', 2: 'Zerg', 3: 'Protoss'}
        returns = []
        for home in range(2):
            away = 1 if home == 0 else 0
            ret = dict()
            ret['map_name'] = info.map_name
            ret['home_race'] = race_dict[info.player_info[home].player_info.race_actual]
            ret['home_result'] = info.player_info[home].player_result.result
            ret['away_race'] = race_dict[info.player_info[away].player_info.race_actual]
            ret['away_result'] = info.player_info[away].player_result.result
            returns.append(ret)
        return returns

    def run(self):
        while True:
            try:
                replay_path = self.replay_queue.get(block=True, timeout=1)
                serial_number = self.serial_queue.get(block=True, timeout=1)
            except queue.Empty:
                return
            try:
                replay_data = self.run_config.replay_data(replay_path)
                version = replay.get_replay_version(replay_data)
                run_config = run_configs.get(version=version)
                handle = run_config.start(want_rgb=False)
                controller = handle.controller
                info = controller.replay_info(replay_data)
                validated_data = self.parse_info(info, replay_path)
                if validated_data is not None:
                    for player in range(PLAYER_NUM):
                        if validated_data[player]['home_race'] == 'Zerg':
                            stat, map_size, born_location = self.replay_decode(
                                controller, replay_path, player, info.game_duration_loops
                            )
                            # z_template = {'beginning_build_order': None, 'cumulative_stat': None, 'map_name': None,
                            #               'map_size': None,
                            #               'born_location': None, 'opponent_born_location': None,
                            #               'home_race': None, 'away_race': None, 'home_result': None}
                            z = dict()
                            z['beginning_build_order'] = stat['begin_statistics']
                            z['cumulative_stat'] = stat['cumulative_statistics']
                            z['map_name'] = info.map_name
                            z['map_size'] = [map_size.x, map_size.y]
                            z['born_location'] = born_location[0]
                            z['opponent_born_location'] = born_location[1]
                            z['home_race'] = validated_data[player]['home_race']
                            z['away_race'] = validated_data[player]['away_race']
                            z['home_result'] = validated_data[player]['home_result']
                            # save data
                            name = '{}_{}_{}_{}_{}_{}'.format(
                                z['home_race'], z['away_race'], z['home_result'], z['map_name'],
                                os.path.basename(replay_path).split('.')[0], str(serial_number)
                            )
                            torch.save(z, os.path.join(self.output_dir, name + '.z'))
                            self.success_msg.put(str(os.getpid()) + ' success parse replay: ' + replay_path)
            except Exception as e:
                self.error_msg.put(repr(e))
            finally:
                try:
                    handle.close()
                except:
                    pass


def print_result():
    data = os.listdir(FLAGS.output_dir)
    print('total:', len(data))
    z = {}
    for i in data:
        a = torch.load(os.path.join(FLAGS.output_dir, i))
        if a['away_race'] == 'Protoss':
            a['away_race'] = 'ZVP'
        if a['away_race'] == 'Terran':
            a['away_race'] = 'ZVT'
        if a['away_race'] == 'Zerg':
            a['away_race'] = 'ZVZ'
        if a['map_name'] not in z.keys():
            z[a['map_name']] = dict()
            z[a['map_name']][a['away_race']] = 1
        elif a['away_race'] not in z[a['map_name']].keys():
            z[a['map_name']][a['away_race']] = 1
        else:
            z[a['map_name']][a['away_race']] += 1
    for i in z.items():
        print(i)


def main(unused_argv):
    print("Getting replay list:", FLAGS.replays)
    replay_list = []
    for root, dirs, files in os.walk(FLAGS.replays):
        for name in files:
            if name.lower().endswith(".sc2replay"):
                replay_list.append(os.path.join(root, name))
    print(len(replay_list), "replays found.")
    random.shuffle(replay_list)
    replay_queue = multiprocessing.Queue()
    serial_queue = multiprocessing.Queue()
    for i in range(len(replay_list)):
        replay_queue.put(replay_list[i])
        serial_queue.put(i)
    decoders = []
    success_msg = multiprocessing.Queue()
    error_msg = multiprocessing.Queue()
    for i in range(PROCESS_NUM):
        decoder = ReplayDecoder(replay_queue, serial_queue, FLAGS.output_dir, success_msg, error_msg)
        decoder.start()
        decoders.append(decoder)
    for i in decoders:
        i.join()
    suc_msg = []
    err_msg = []
    while not success_msg.empty():
        suc_msg.append(success_msg.get())
    while not error_msg.empty():
        err_msg.append(error_msg.get())

    def log_func(s, idx):
        return s + '\n{}\t'.format(idx) + '-' * 60 + '\n'

    new_success_msg = [log_func(s, idx) for idx, s in enumerate(suc_msg)]
    new_error_msg = [log_func(s, idx) for idx, s in enumerate(err_msg)]
    with open(os.path.join(FLAGS.output_dir, 'success.txt'), 'w') as f:
        f.writelines(new_success_msg)
    with open(os.path.join(FLAGS.output_dir, 'error.txt'), 'w') as f:
        f.writelines(new_error_msg)


if __name__ == "__main__":
    app.run(main)
