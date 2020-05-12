# coding: utf-8
'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Generate torch-style data for supervised learning from replays

All data in proto format refers to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/sc2api.proto
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import platform
import signal
import sys
import collections
import time
import copy
import traceback

from absl import app
from absl import logging
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import torch
import numpy as np
from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import replay
from pysc2.lib.remote_controller import RequestError

from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser, compress_obs, decompress_obs
from sc2learner.envs.observations import get_enemy_upgrades_raw_data, get_enemy_upgrades_processed_data
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser, remove_repeat_data
from sc2learner.envs.statistics import RealTimeStatistics, transform_stat, transform_cum_stat
from sc2learner.envs.maps.map_info import LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT
from sc2learner.utils import save_file_ceph

logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "D:/game/replays", "Path to a directory of replays.")
flags.DEFINE_string("output_dir", "E:/data/replay_data_test", "Path to save data")
flags.DEFINE_string("version", "4.10.0", "Game version")
flags.DEFINE_integer("process_num", 1, "Number of sc2 process to start on the node")
flags.DEFINE_boolean("crop", False, "whether to crop the map to playable area")
flags.DEFINE_bool(
    "check_version", True, "Check required game version of the replays and discard ones not matching version"
)
flags.DEFINE_bool("resolution", False, "whether to use defined resolution rather than map size as spatial size")
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("output_dir")
flags.FLAGS(sys.argv)

FeatureUnit = features.FeatureUnit
Action = collections.namedtuple('Action', ['action', 'game_loop'])
RESOLUTION = 128


class ReplayDecoder(multiprocessing.Process):
    def __init__(self, run_config, replay_list, output_dir, ues_resolution, crop):
        super(ReplayDecoder, self).__init__()
        self.run_config = run_config
        self.replay_list = replay_list
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass
        self.output_dir = output_dir
        self.obs_parser = AlphastarObsParser()
        size = point.Point(1, 1)
        self.interface = sc_pb.InterfaceOptions(
            raw=True,
            score=False,
            raw_crop_to_playable_area=crop,
            feature_layer=sc_pb.SpatialCameraSetup(width=24, crop_to_playable_area=crop)
        )
        size.assign_to(self.interface.feature_layer.resolution)
        size.assign_to(self.interface.feature_layer.minimap_resolution)
        self.use_resolution = ues_resolution
        self.crop = crop

    def replay_decode(self, replay_data, game_loops):
        """Where real decoding is happening"""
        PLAYER_NUM = 2
        map_size = None
        # get actions first
        actions = []
        minimum_loop = game_loops  # cut off invalid winner actions
        cut_player = 0  # default winner
        map_size_point = point.Point(1, 1)  # action doesn't need spatial information, for speeding up
        map_size_point.assign_to(self.interface.feature_layer.minimap_resolution)
        for player in range(PLAYER_NUM):
            logging.info('Start getting game loops for player {}'.format(player))
            self.controller.start_replay(
                sc_pb.RequestStartReplay(
                    replay_data=replay_data,
                    options=self.interface,
                    observed_player_id=player + 1,
                )
            )
            map_size = self.controller.game_info().start_raw.map_size
            assert [map_size.x, map_size.y] == self.map_size, 'map crop failed, check gcc version!'
            player_actions = []
            cur_loop = 0
            while cur_loop < game_loops:
                next_loop = min(game_loops, cur_loop + 1000)
                self.controller.step(next_loop - cur_loop)
                cur_loop = next_loop
                ob = self.controller.observe()
                for i in ob.actions:
                    if i.HasField('action_raw'):
                        if not i.action_raw.HasField('camera_move'):
                            assert i.HasField('game_loop')  # debug
                            action = Action(i.action_raw, i.game_loop)
                            player_actions.append(action)
            if player_actions[-1].game_loop <= minimum_loop:
                assert (player_actions[-1].game_loop != 0)  # valid game_loop
                minimum_loop = player_actions[-1].game_loop
                cut_player = 1 - player
            actions.append(player_actions)
        for i, action in enumerate(actions[cut_player]):  # cut actions
            if action.game_loop > minimum_loop:
                actions[cut_player] = actions[cut_player][:i]
                break

        def unit_id_mapping(obs):
            raw_units = obs['raw_units']
            key_index = FeatureUnit['unit_type']
            # TODO: Why this vvvv
            for i in range(raw_units.shape[0]):
                if raw_units[i, key_index] == 1879:
                    raw_units[i, key_index] = 1904
                elif raw_units[i, key_index] == 1883:
                    raw_units[i, key_index] = 1908
            return obs

        def check_step(obs, action):
            tags = set()
            selected_tags = set()
            for i in obs.observation.raw_data.units:
                tags.add(i.tag)
            if action.HasField('unit_command'):
                uc = action.unit_command
                for i in uc.unit_tags:
                    selected_tags.add(i)
                if uc.HasField('target_unit_tag'):
                    target_tag = uc.target_unit_tag
                    if target_tag not in tags:
                        return False
                if len(selected_tags - tags) > 0:
                    return False
            return True

        # view replay by action order, combine observation and action
        step_data = [[] for _ in range(PLAYER_NUM)]  # initial return data
        stat = [RealTimeStatistics(begin_num=20) for _ in range(PLAYER_NUM)]
        enemy_upgrades = [None for _ in range(PLAYER_NUM)]
        born_location = [[] for _ in range(PLAYER_NUM)]
        for player in range(PLAYER_NUM):  # gain data by player order
            logging.info('Start getting data for player {}'.format(player))

            assert map_size is not None
            map_size_point = point.Point(map_size.x, map_size.y)
            if self.use_resolution:
                resolution = point.Point(RESOLUTION, RESOLUTION)
                resolution.assign_to(self.interface.feature_layer.minimap_resolution)
            else:
                map_size_point.assign_to(self.interface.feature_layer.minimap_resolution)  # update map size
            self.controller.start_replay(
                sc_pb.RequestStartReplay(
                    replay_data=replay_data, options=self.interface, observed_player_id=player + 1
                )
            )
            ob = self.controller.observe()
            # get self born location
            location = []
            for i in ob.observation.raw_data.units:
                if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
                    location.append([i.pos.x, i.pos.y])
            assert len(location) == 1, 'this replay is corrupt, no fog of war, check replays from this game version'
            born_location[player] = location[0]
            feat = features.features_from_game_info(self.controller.game_info(), use_raw_actions=True)
            ob = feat.transform_obs(ob)
            assert np.sum(ob['feature_minimap']['buildable']) > 0, 'no buildable map, check gcc version!'
            act_parser = AlphastarActParser(
                feature_layer_resolution=RESOLUTION, map_size=map_size_point, use_resolution=self.use_resolution
            )
            actions[player].append(Action(None, game_loops + 1))  # padding
            last_action = {
                'action_type': torch.LongTensor([0]),
                'delay': torch.LongTensor([0]),
                'queued': 'none',
                'selected_units': 'none',
                'target_units': 'none',
                'target_location': 'none'
            }
            self.controller.step(actions[player][0].game_loop)
            last_print = 0
            for idx, action in enumerate(actions[player]):
                if idx == len(actions[player]) - 1:
                    break
                obs = self.controller.observe()
                delay = actions[player][idx + 1].game_loop - action.game_loop  # game_steps between two actions
                if not check_step(obs, action.action):
                    last_action['delay'] += delay
                    step_data[player][-1]['actions']['delay'] += delay
                    self.controller.step(delay)
                    continue
                assert (obs.observation.game_loop == action.game_loop)
                if action.game_loop - last_print > 1000:
                    last_print = action.game_loop
                    logging.info('Progress ({}){:5}/{:5}'.format(player, action.game_loop, game_loops))
                base_ob = feat.transform_obs(obs)
                base_ob = unit_id_mapping(base_ob)
                agent_ob = self.obs_parser.parse(base_ob)
                agent_act = act_parser.parse(action.action)
                assert len(agent_act) == 1
                agent_act = act_parser.merge_same_id_action(agent_act)[0]
                agent_act['delay'] = torch.LongTensor([delay])
                agent_ob['scalar_info']['cumulative_stat'] = transform_cum_stat(stat[player].cumulative_statistics)
                stat[player].update_stat(agent_act, agent_ob, action.game_loop)
                result_obs = self.obs_parser.merge_action(agent_ob, last_action, True)
                # get_enemy_upgrades_processed_data must be used after merge_action
                # enemy_upgrades_raw = get_enemy_upgrades_raw_data(base_ob, copy.deepcopy(enemy_upgrades[player]))
                enemy_upgrades_proc = get_enemy_upgrades_processed_data(result_obs, enemy_upgrades[player])
                result_obs['scalar_info']['enemy_upgrades'] = enemy_upgrades_proc
                enemy_upgrades[player] = enemy_upgrades_proc
                result_obs.update({'actions': agent_act})
                # store only the compressed obs, and let gc clear uncompressed obs
                compressed_obs = copy.deepcopy(compress_obs(result_obs))
                step_data[player].append(compressed_obs)
                last_action = agent_act
                self.controller.step(delay)
        return (
            step_data, [stat[i].get_stat() for i in range(PLAYER_NUM)], map_size,
            [stat[i].cumulative_statistics_game_loop for i in range(PLAYER_NUM)], born_location
        )

    def parse_info(self, info, replay_path):
        if (info.HasField("error")):
            logging.warning('Info have error')
            return None
        if (info.map_name not in LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT.keys()):
            logging.error(LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT.keys())
            logging.error(
                'Found replay using unknown map {}, or there is sth wrong with locale'.format(info.map_name) +
                ' Try regenerate map_info.py'
            )
            return None
        if ('.'.join(info.game_version.split('.')[:3]) != self.run_config.version.game_version):
            logging.warning('Wrong version')
            return None
        if info.game_duration_loops < 1000:
            logging.warning('Game too short:{}'.format(info.game_duration_loops))
            return None
        if len(info.player_info) != 2:
            logging.warning('len(player_info)={}'.format(len(info.player_info)))
            return None
        for p in info.player_info:
            if (p.HasField('player_apm') and p.player_apm < 10
                    or (p.HasField('player_mmr') and p.player_mmr < 3500)):  # noqa
                # Low APM = player just standing around.
                # Low MMR = corrupt replay or player who is weak.
                logging.warning('Low APM or MMR')
                return None
        if (info.player_info[0].player_info.race_actual != 2 and info.player_info[1].player_info.race_actual != 2):
            # not include Zerg race
            logging.warning('No Zerg')
            return None
        race_dict = {1: 'Terran', 2: 'Zerg', 3: 'Protoss'}
        returns = []
        for home in range(2):
            away = 1 if home == 0 else 0
            ret = dict()
            ret['game_duration_loops'] = info.game_duration_loops
            ret['game_version'] = info.game_version
            ret['map_name'] = LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT[info.map_name]
            ret['home_race'] = race_dict[info.player_info[home].player_info.race_actual]
            ret['home_mmr'] = info.player_info[home].player_mmr
            ret['home_apm'] = info.player_info[home].player_apm
            ret['home_result'] = info.player_info[home].player_result.result
            ret['away_race'] = race_dict[info.player_info[away].player_info.race_actual]
            ret['away_mmr'] = info.player_info[away].player_mmr
            ret['away_apm'] = info.player_info[away].player_apm
            ret['away_result'] = info.player_info[away].player_result.result
            ret['replay_path'] = replay_path
            ret['screen_resolution'] = 1  # placeholder
            returns.append(ret)
        return returns

    def run(self):
        # interface to be called when starting process
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())
        self.handle = self.run_config.start(want_rgb=False)
        self.controller = self.handle.controller
        print('Subprocess {}: Started successfully.'.format(os.getpid()))
        for replay_path in self.replay_list:
            t = time.time()
            try:
                logging.info('Start working on {}'.format(replay_path))
                replay_data = self.run_config.replay_data(replay_path)
                info = self.controller.replay_info(replay_data)
                validated_data = self.parse_info(info, replay_path)
                if validated_data is not None:
                    self.map_size = get_map_size(validated_data[0]['map_name'], cropped=self.crop)
                    (step_data, stat, map_size, cumulative_z,
                     born_location) = self.replay_decode(replay_data, info.game_duration_loops)
                    validated_data[0]['map_size'] = [map_size.x, map_size.y]
                    validated_data[1]['map_size'] = [map_size.x, map_size.y]
                    meta_data_0 = validated_data[0]
                    meta_data_1 = validated_data[1]
                    # remove repeat data
                    # step_data = [remove_repeat_data(d) for d in step_data]
                    meta_data_0['step_num'] = len(step_data[0])
                    meta_data_1['step_num'] = len(step_data[1])
                    # transform stat
                    stat_processed_0 = transform_stat(stat[0], meta_data_0)
                    stat_processed_1 = transform_stat(stat[1], meta_data_1)
                    # modify Z from this replay
                    # z_template = {'beginning_build_order': None, 'cumulative_stat': None, 'map_name': None,
                    #               'map_size': None, 'home_mmr': None,
                    #               'born_location': None, 'opponent_born_location': None,
                    #               'home_race': None, 'away_race': None, 'home_result': None}
                    stat_z = [{} for _ in range(2)]
                    for i in range(2):
                        stat_z[i]['beginning_build_order'] = stat[i]['begin_statistics']
                        stat_z[i]['cumulative_stat'] = cumulative_z[i]
                        stat_z[i]['map_name'] = validated_data[i]['map_name']
                        stat_z[i]['map_size'] = [map_size.x, map_size.y]
                        stat_z[i]['home_mmr'] = validated_data[i]['home_mmr']
                        stat_z[i]['born_location'] = born_location[i]
                        stat_z[i]['opponent_born_location'] = born_location[1 - i]
                        stat_z[i]['home_race'] = validated_data[i]['home_race']
                        stat_z[i]['away_race'] = validated_data[i]['away_race']
                        stat_z[i]['home_result'] = validated_data[i]['home_result']
                    # save data
                    name0 = '{}_{}_{}_{}'.format(
                        meta_data_0['home_race'], meta_data_0['away_race'], meta_data_0['home_mmr'],
                        os.path.basename(replay_path).split('.')[0]
                    )
                    name1 = '{}_{}_{}_{}'.format(
                        meta_data_1['home_race'], meta_data_1['away_race'], meta_data_1['home_mmr'],
                        os.path.basename(replay_path).split('.')[0]
                    )
                    # torch.save(meta_data_0, os.path.join(self.output_dir, name0 + '.meta'))
                    # torch.save(step_data[0], os.path.join(self.output_dir, name0 + '.step'))
                    # torch.save(stat[0], os.path.join(self.output_dir, name0 + '.stat'))
                    # torch.save(stat_processed_0, os.path.join(self.output_dir, name0 + '.stat_processed'))
                    # torch.save(stat_z[0], os.path.join(self.output_dir, name0 + '.z'))
                    # torch.save(meta_data_1, os.path.join(self.output_dir, name1 + '.meta'))
                    # torch.save(step_data[1], os.path.join(self.output_dir, name1 + '.step'))
                    # torch.save(stat[1], os.path.join(self.output_dir, name1 + '.stat'))
                    # torch.save(stat_processed_1, os.path.join(self.output_dir, name1 + '.stat_processed'))
                    # torch.save(stat_z[1], os.path.join(self.output_dir, name1 + '.z'))

                    ceph_root = 's3://replay_decode_493_2'
                    save_file_ceph(ceph_root, name0 + '.meta', meta_data_0)
                    save_file_ceph(ceph_root, name0 + '.step', step_data[0])
                    save_file_ceph(ceph_root, name0 + '.stat', stat[0])
                    save_file_ceph(ceph_root, name0 + '.stat_processed', stat_processed_0)
                    save_file_ceph(ceph_root, name0 + '.z', stat_z[0])
                    save_file_ceph(ceph_root, name1 + '.meta', meta_data_1)
                    save_file_ceph(ceph_root, name1 + '.step', step_data[1])
                    save_file_ceph(ceph_root, name1 + '.stat', stat[1])
                    save_file_ceph(ceph_root, name1 + '.stat_processed', stat_processed_1)
                    save_file_ceph(ceph_root, name1 + '.z', stat_z[1])

                    logging.info(
                        'Replay parsing success, t=({}) ({})({}): {}'.format(
                            time.time() - t, str(meta_data_0), str(meta_data_1), replay_path
                        )
                    )
                else:
                    logging.warning('Invalid replay: ' + replay_path)
            except RequestError as e:
                logging.error('SC2 RequestError:' + str(e))
                self.handle.close()
                self.handle = self.run_config.start(want_rgb=False)
                self.controller = self.handle.controller
            except Exception as e:
                logging.info(''.join(traceback.format_tb(e.__traceback__)))
                logging.info('InnerError: {}'.format(sys.exc_info()))
        self.handle.close()


def main(unused_argv):
    os.environ['SC2PATH'] = '/mnt/lustre/zhangming/StarCraftII_Linux_Package/StarCraftII_' + FLAGS.version
    print(os.environ['SC2PATH'])
    run_config = run_configs.get(FLAGS.version)
    if platform.system() == 'Windows':
        replay_list = sorted(run_config.replay_paths(FLAGS.replays))
    else:
        print('processing {}'.format(FLAGS.replays))
        replay_list = [x.strip() for x in open(FLAGS.replays, 'r').readlines()]
    fitered_replays = []  # filter replays by version
    if FLAGS.check_version:
        for replay_path in replay_list:
            version = replay.get_replay_version(run_config.replay_data(replay_path))
            if version.game_version == FLAGS.version:
                fitered_replays.append(replay_path)
    else:
        fitered_replays = replay_list
    N = FLAGS.process_num
    if N > 1:
        # using multiprocessing
        group_num = int(len(fitered_replays) // N)
        print('Total len: {}, group: {}, each group: {}'.format(len(fitered_replays), N, group_num))
        # ISSUE(zh) splited group number doesn't match pool
        replay_split_list = [fitered_replays[i * group_num:(i + 1) * group_num] for i in range(N)]
        decoders = []
        print('Writing output to: {}'.format(FLAGS.output_dir))
        for i in range(N):
            decoder = ReplayDecoder(run_config, replay_split_list[i], FLAGS.output_dir, FLAGS.resolution, FLAGS.crop)
            decoder.start()
            decoders.append(decoder)
        for i in decoders:
            i.join()
    else:
        # single process
        decoder = ReplayDecoder(run_config, fitered_replays, FLAGS.output_dir, FLAGS.resolution, FLAGS.crop)
        decoder.run()


if __name__ == "__main__":
    app.run(main)
