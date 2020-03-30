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
import signal
import sys
import collections
import time
import copy

from absl import app
from absl import logging
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import torch

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import replay
from pysc2.lib.remote_controller import RequestError

from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser, compress_obs, decompress_obs, \
    transform_cum_stat, transform_stat
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser, remove_repeat_data
from sc2learner.envs.maps.map_info import LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT

logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "E:/data/replays", "Path to a directory of replays.")
flags.DEFINE_string("output_dir", "E:/data/replay_data", "Path to save data")
flags.DEFINE_string("version", "4.10.0", "Game version")
flags.DEFINE_integer("process_num", 1, "Number of sc2 process to start on the node")
flags.DEFINE_bool("check_version", False,
    "Check required game version of the replays and discard ones not matching version")
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("output_dir")
flags.FLAGS(sys.argv)

FeatureUnit = features.FeatureUnit
Action = collections.namedtuple('Action', ['action', 'game_loop'])


class ReplayDecoder(multiprocessing.Process):
    def __init__(self, run_config, replay_list, output_dir):
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
        self.interface = sc_pb.InterfaceOptions(raw=True, score=False, feature_layer=sc_pb.SpatialCameraSetup(width=24))
        size.assign_to(self.interface.feature_layer.resolution)
        size.assign_to(self.interface.feature_layer.minimap_resolution)

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
            player_actions = []
            cur_loop = 0
            while cur_loop < game_loops:
                next_loop = min(game_loops, cur_loop + 1000)
                self.controller.step(next_loop - cur_loop)
                cur_loop = next_loop
                ob = self.controller.observe()
                for i in ob.actions:
                    if i.HasField('action_raw'):
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

        def update_action_stat(action_statistics, act, obs):
            def get_unit_types(units, entity_type_dict):
                unit_types = set()
                for u in units:
                    try:
                        unit_type = entity_type_dict[u]
                        unit_types.add(unit_type)
                    except KeyError:
                        logging.warning("Not found unit(id: {})".format(u))
                return unit_types

            action_type = act['action_type'].item()
            if action_type not in action_statistics.keys():
                action_statistics[action_type] = {
                    'count': 0,
                    'selected_type': set(),
                    'target_type': set(),
                }
            action_statistics[action_type]['count'] += 1
            entity_type_dict = {id: type for id, type in zip(obs['entity_raw']['id'], obs['entity_raw']['type'])}
            if isinstance(act['selected_units'], torch.Tensor):
                units = act['selected_units'].tolist()
                unit_types = get_unit_types(units, entity_type_dict)
                action_statistics[action_type]['selected_type'] = action_statistics[action_type]['selected_type'].union(
                    unit_types
                )  # noqa
            if isinstance(act['target_units'], torch.Tensor):
                units = act['target_units'].tolist()
                unit_types = get_unit_types(units, entity_type_dict)
                action_statistics[action_type]['target_type'] = action_statistics[action_type]['target_type'].union(
                    unit_types
                )  # noqa

        def update_cum_stat(cumulative_statistics, act):
            action_type = act['action_type'].item()
            goal = GENERAL_ACTION_INFO_MASK[action_type]['goal']
            if goal != 'other':
                if action_type not in cumulative_statistics.keys():
                    cumulative_statistics[action_type] = {'count': 1, 'goal': goal}
                else:
                    cumulative_statistics[action_type]['count'] += 1

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

        def unit_record(obs, s):
            for idx, ob in enumerate(obs):
                raw_set = ob['entity_raw']['id']
                # units = [4352376833]  # for debug
                # for u in units:
                #    if u in raw_set:
                #        i = raw_set.index(u)
                #        print(u, idx, ob['entity_raw']['type'][i], s)

        # view replay by action order, combine observation and action
        step_data = [[] for _ in range(PLAYER_NUM)]  # initial return data
        action_statistics = [{} for _ in range(PLAYER_NUM)]
        cumulative_statistics = [{} for _ in range(PLAYER_NUM)]
        begin_statistics = [[] for _ in range(PLAYER_NUM)]
        begin_num = 200
        for player in range(PLAYER_NUM):  # gain data by player order
            logging.info('Start getting data for player {}'.format(player))
            assert map_size is not None
            map_size_point = point.Point(map_size.x, map_size.y)
            map_size_point.assign_to(self.interface.feature_layer.minimap_resolution)  # update map size
            self.controller.start_replay(
                sc_pb.RequestStartReplay(
                    replay_data=replay_data, options=self.interface, observed_player_id=player + 1
                )
            )
            feat = features.features_from_game_info(self.controller.game_info(), use_raw_actions=True)
            act_parser = AlphastarActParser(feature_layer_resolution=1, map_size=map_size_point)
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
                else:
                    delay = actions[player][idx + 1].game_loop - action.game_loop  # game_steps between two actions
                obs = self.controller.observe()
                assert (obs.observation.game_loop == action.game_loop)
                if action.game_loop - last_print > 1000:
                    last_print = action.game_loop
                    logging.info('Progress ({}){:5}/{:5}'.format(player, action.game_loop, game_loops))
                base_ob = feat.transform_obs(obs)
                base_ob = unit_id_mapping(base_ob)
                agent_ob = self.obs_parser.parse(base_ob)
                # TODO(zh) compute enemy_upgrades by observation
                agent_ob['scalar_info']['enemy_upgrades'] = torch.zeros(128, dtype=torch.long)
                agent_act = act_parser.parse(action.action)
                assert len(agent_act) == 1
                agent_act = act_parser.merge_same_id_action(agent_act)[0]
                agent_act['delay'] = torch.LongTensor([delay])
                update_action_stat(action_statistics[player], agent_act, agent_ob)
                update_cum_stat(cumulative_statistics[player], agent_act)
                if len(begin_statistics[player]) < begin_num:
                    update_begin_stat(begin_statistics[player], agent_act)
                agent_ob['scalar_info']['cumulative_stat'] = transform_cum_stat(cumulative_statistics[player])
                result_obs = self.obs_parser.merge_action(agent_ob, last_action, True)
                result_obs.update({'actions': agent_act})
                # store only the compressed obs, and let gc clear uncompressed obs
                compressed_obs = copy.deepcopy(compress_obs(result_obs))
                step_data[player].append(compressed_obs)
                last_action = agent_act
                self.controller.step(delay)
        return (
            step_data, [
                {
                    'action_statistics': action_statistics[idx],
                    'cumulative_statistics': cumulative_statistics[idx],
                    'begin_statistics': begin_statistics[idx]
                } for idx in range(2)
            ], map_size
        )

    def parse_info(self, info, replay_path):
        if (info.HasField("error")):
            logging.warning('Info have error')
            return None
        if(info.map_name not in LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT.keys()):
            logging.error('Found replay using unknown map {}, or there is sth wrong with locale'
                           .format(info.map_name) + ' Try regenerate map_info.py')
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
                    or (p.HasField('player_mmr') and p.player_mmr < 1000)):  # noqa
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
                    step_data, stat, map_size = self.replay_decode(replay_data, info.game_duration_loops)
                    validated_data[0]['map_size'] = [map_size.x, map_size.y]
                    validated_data[1]['map_size'] = [map_size.x, map_size.y]
                    meta_data_0 = validated_data[0]
                    meta_data_1 = validated_data[1]
                    # remove repeat data
                    step_data = [remove_repeat_data(d) for d in step_data]
                    meta_data_0['step_num'] = len(step_data[0])
                    meta_data_1['step_num'] = len(step_data[1])
                    # transform stat
                    stat_processed_0 = transform_stat(stat[0], meta_data_0)
                    stat_processed_1 = transform_stat(stat[1], meta_data_1)
                    # save data
                    name0 = '{}_{}_{}_{}'.format(
                        meta_data_0['home_race'], meta_data_0['away_race'], meta_data_0['home_mmr'],
                        os.path.basename(replay_path).split('.')[0]
                    )
                    name1 = '{}_{}_{}_{}'.format(
                        meta_data_1['home_race'], meta_data_1['away_race'], meta_data_1['home_mmr'],
                        os.path.basename(replay_path).split('.')[0]
                    )
                    torch.save(meta_data_0, os.path.join(self.output_dir, name0 + '.meta'))
                    torch.save(step_data[0], os.path.join(self.output_dir, name0 + '.step'))
                    torch.save(stat[0], os.path.join(self.output_dir, name0 + '.stat'))
                    torch.save(stat_processed_0, os.path.join(self.output_dir, name0 + '.stat_processed'))
                    torch.save(meta_data_1, os.path.join(self.output_dir, name1 + '.meta'))
                    torch.save(step_data[1], os.path.join(self.output_dir, name1 + '.step'))
                    torch.save(stat[1], os.path.join(self.output_dir, name1 + '.stat'))
                    torch.save(stat_processed_1, os.path.join(self.output_dir, name1 + '.stat_processed'))
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
        self.handle.close()


def main(unused_argv):
    run_config = run_configs.get(FLAGS.version)
    replay_list = sorted(run_config.replay_paths(FLAGS.replays))
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
            decoder = ReplayDecoder(run_config, replay_split_list[i], FLAGS.output_dir)
            decoder.start()
            decoders.append(decoder)
        for i in decoders:
            i.join()
    else:
        # single process
        decoder = ReplayDecoder(run_config, fitered_replays, FLAGS.output_dir)
        decoder.run()


if __name__ == "__main__":
    app.run(main)
