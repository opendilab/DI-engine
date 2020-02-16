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
import time
import os
import signal
import sys

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import torch

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from pysc2.lib import replay

from pysc2.lib import gfile
from pysc2.lib.action_dict import ACTION_INFO_MASK
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser, compress_obs, decompress_obs
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser
from functools import partial

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 1, "How many game steps per observation.")
flags.DEFINE_string("replays", None, "Path to a directory of replays.")
flags.DEFINE_string("output_dir", "/mnt/lustre/niuyazhe/data/sl_data_test", "Path to save data")
flags.mark_flag_as_required("replays")


RESOLUTION = 128
FeatureUnit = features.FeatureUnit
size = point.Point(RESOLUTION, RESOLUTION)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)


def sorted_dict_str(d):
    return "{%s}" % ", ".join("%s: %s" % (k, d[k])
                              for k in sorted(d, key=d.get, reverse=True))


def valid_replay(info, ping):
    '''
        Overview: Make sure the replay isn't corrupt, and is worth looking at
        Arguments:
            - info (:obj:'ResponseReplayInfo'): replay information in proto format
            - ping (:obj:'ResponsePing'): debug information for version check in proto format
        Returns:
            - (:obj'bool'): the replay is valid or not
    '''
    if (info.HasField("error") or
        info.base_build != ping.base_build or  # different game version
        info.game_duration_loops < 1000 or
            len(info.player_info) != 2):
        # Probably corrupt, or just not interesting.
        return False
    for p in info.player_info:
        if p.player_apm < 10 or p.player_mmr < 1000:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return False
    return True


class ReplayProcessor(multiprocessing.Process):
    '''
        Overview: a process decodes a single replay
        Interface: __init__, run
    '''

    def __init__(self, run_config, output_dir=None):
        '''
            Overview: parse run_config and prepare related attributes
            Arguments:
                - run_config (:obj：'RunConfig'): starcraft2 run config
                - output_dir (:obj：'string'): path to save data
        '''
        super(ReplayProcessor, self).__init__()
        assert(output_dir is not None)
        self.run_config = run_config
        self.output_dir = output_dir
        self.obs_parser = AlphastarObsParser()
        self.handles = []
        self.controllers = []
        self.player_ids = [i+1 for i in range(2)]
        # start game and initial two game controlloers for both players, controller hanldes communication with game
        for i in self.player_ids:
            handle = self.run_config.start(want_rgb=interface.HasField("render"))
            controller = handle.controller
            self.handles.append(handle)
            self.controllers.append(controller)
        self._print("SC2 Started successfully.")

    def _replay_prepare(self, controller, replay_path, print_info=True):
        '''
            Overview: get basic information from a replay and validate it 
            Arguments:
                - controller (:obj:'RemoteController'): game controller whick takes actions and generates observations in proto format
                - replay_path (:obj:'string'): path to the replay
                - print_info (:obj:'bool'): a bool decides whether to print replay info
            Returns:
                - replay_data (:obj'bytes'): replay file
                - map_data (:obj'bytes'): map file
                - info (:obj'ResponseReplayInfo'): replay information in proto format
        '''
        ping = controller.ping()
        replay_name = os.path.basename(replay_path)[:10]
        replay_data = self.run_config.replay_data(replay_path)
        info = controller.replay_info(replay_data)
        if print_info:
            self._print("Got replay: %s" % replay_path)
            self._print((" Replay Info %s " % replay_name).center(60, "-"))
            self._print(info)
            self._print("-" * 60)
        if valid_replay(info, ping):
            map_data = None
            if info.local_map_path:
                map_data = self.run_config.map_data(info.local_map_path)
            return replay_data, map_data, info
        else:
            self._print("Replay is invalid.")
            return None

    def _parse_info(self, info, replay_path, home=0):
        '''
            Overview: parse replay basic information into a dict
            Arguments:
                - info (:obj:'ResponseReplayInfo'): reaplay information in proto format
                - replay_path (:obj:'string'): path to the replay
                - home (:obj:'int'): player id to identify player
            Returns:
                - ret (:obj'dict'): replay information dict 
        '''
        away = 1 if home == 0 else 0
        race_dict = {1: 'Terran', 2: 'Zerg', 3: 'Protoss'}
        ret = {}
        ret['game_duration_loops'] = info.game_duration_loops
        ret['game_version'] = info.game_version
        ret['map_name'] = info.map_name
        ret['home_race'] = race_dict[info.player_info[home].player_info.race_actual]
        ret['home_mmr'] = info.player_info[home].player_mmr
        ret['home_apm'] = info.player_info[home].player_apm
        ret['home_result'] = info.player_info[home].player_result.result
        ret['away_race'] = race_dict[info.player_info[away].player_info.race_actual]
        ret['away_mmr'] = info.player_info[away].player_mmr
        ret['away_apm'] = info.player_info[away].player_apm
        ret['away_result'] = info.player_info[away].player_result.result
        ret['replay_path'] = replay_path
        ret['feature_layer_resolution'] = RESOLUTION
        return ret

    def run(self, replay_path):
        '''
            Overview: run a ReplayProcessor and save data
            Returns:
                - (:obj'string'): replay parse result plus replay path
        '''
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        self._print("Starting up a new SC2 instance.")
        try:
            ret = self._replay_prepare(self.controllers[0], replay_path)
            if ret is not None:
                replay_data, map_data, info = ret
                meta_data_0 = self._parse_info(info, replay_path, home=0)
                meta_data_1 = self._parse_info(info, replay_path, home=1)
                step_data, map_size, stat = self.process_replay_multi(
                    self.controllers, replay_data, map_data, self.player_ids)
                meta_data_0['step_num'] = len(step_data[0])
                meta_data_1['step_num'] = len(step_data[1])
                meta_data_0['map_size'] = map_size
                meta_data_1['map_size'] = map_size
                name0 = '{}_{}_{}_{}'.format(meta_data_0['home_race'], meta_data_0['away_race'],
                                             meta_data_0['home_mmr'], os.path.basename(replay_path).split('.')[0])
                name1 = '{}_{}_{}_{}'.format(meta_data_1['home_race'], meta_data_1['away_race'],
                                             meta_data_1['home_mmr'], os.path.basename(replay_path).split('.')[0])
                torch.save(meta_data_0, os.path.join(self.output_dir, name0+'.meta'))
                torch.save(step_data[0], os.path.join(self.output_dir, name0+'.step'))
                torch.save(stat[0], os.path.join(self.output_dir, name0+'.stat'))
                torch.save(meta_data_1, os.path.join(self.output_dir, name1+'.meta'))
                torch.save(step_data[1], os.path.join(self.output_dir, name1+'.step'))
                torch.save(stat[1], os.path.join(self.output_dir, name1+'.stat'))
                return "success parse replay " + replay_path
            else:
                return "invalid replay " + replay_path
        except (protocol.ConnectionError, protocol.ProtocolError,
                remote_controller.RequestError):
            raise Exception
        except KeyboardInterrupt:
            return

    def close(self):
        for handle in self.handles:
            handle.close()

    def _print(self, s):
        for line in str(s).strip().splitlines():
            print("[%s] %s" % (0, line))

    def process_replay_multi(self, controllers, replay_data, map_data, player_ids):
        '''
            Overview: decode a replay step by step to generate data
            Arguments: 
                - controllers (:obj:'RemoteController'): game controller whick takes actions and generates observations in proto format
                - replay_data (:obj:'bytes'): replay file
                - map_data (:obj:'bytes'): map file
                - player_ids (:obj:'int'): player id to identify player
            Returns:
                - step_data (:obj'list'): step data for both players, includes observations and actions
                - map_size (:obj'Size2DI'): map size in proto format
                - stat (:obj'list'): statistics in the replay for both players
        '''
        feats = []
        # start replay with specific settings in both controllers
        for controller, player_id in zip(controllers, player_ids):
            controller.start_replay(sc_pb.RequestStartReplay(
                replay_data=replay_data,
                map_data=map_data,
                options=interface,
                observed_player_id=player_id))
            
            feat = features.features_from_game_info(controller.game_info())
            feats.append(feat)

            controller.step()
        map_size = controllers[0].game_info().start_raw.map_size
        # initial an act_parser to parse actions
        act_parser = AlphastarActParser(feature_layer_resolution=RESOLUTION, map_size=map_size)

        def update_action_stat(action_statistics, act, obs):
            def get_unit_type(tag, obs):
                for idx, v in enumerate(obs["raw_units"][:, FeatureUnit.tag]):
                    if tag == v:
                        return obs["raw_units"][idx, FeatureUnit.unit_type]
                return None

            action_type = act['action_type'].item()
            if action_type not in action_statistics.keys():
                action_statistics[action_type] = {
                    'count': 0,
                    'selected_type': set(),
                    'target_type': set(),
                }
            action_statistics[action_type]['count'] += 1
            if isinstance(act['selected_units'], torch.Tensor):
                for unit_tag in act['selected_units']:
                    unit_type = get_unit_type(unit_tag.item(), obs)
                    if unit_type is None:
                        print("not found selected unit(id: {})".format(unit_tag.item()))
                        return False
                    action_statistics[action_type]['selected_type'].add(unit_type)
            if isinstance(act['target_units'], torch.Tensor):
                for unit_tag in act['target_units']:
                    unit_type = get_unit_type(unit_tag.item(), obs)
                    if unit_type is None:
                        print("not found target unit(id: {})".format(unit_tag.item()))
                        return False
                    action_statistics[action_type]['target_type'].add(unit_type)
            return True

        def update_cum_stat(cumulative_statistics, act):
            action_type = act['action_type'].item()
            goal = ACTION_INFO_MASK[action_type]['goal']
            if goal != 'other':
                if action_type not in cumulative_statistics.keys():
                    cumulative_statistics[action_type] = {'count': 1, 'goal': goal}
                else:
                    cumulative_statistics[action_type]['count'] += 1

        def update_begin_stat(begin_statistics, act):
            target_list = ['unit', 'build', 'research', 'effect']
            action_type = act['action_type'].item()
            goal = ACTION_INFO_MASK[action_type]['goal']
            if goal in target_list:
                if goal == 'build':
                    location = act['target_location']
                    if isinstance(location, torch.Tensor):  # for build ves, no target_location
                        location = location.tolist()
                else:
                    location = 'none'
                begin_statistics.append({'action_type': action_type, 'location': location})

        N = len(player_ids)
        step = 0
        delay = [0 for _ in range(N)]
        action_count = [0 for _ in range(N)]
        last_actions = [{'action_type': torch.LongTensor([0]), 'delay': torch.LongTensor([0]),
                         'queued': 'none', 'selected_units': 'none', 'target_units': 'none',
                         'target_location': 'none'} for _ in range(N)]
        step_data = [[] for _ in range(N)]
        error_set = set()
        action_statistics = [{} for _ in range(N)]
        cumulative_statistics = [{} for _ in range(N)]
        begin_statistics = [[] for _ in range(N)]
        begin_num = 100

        prev_obs = [controller.observe() for controller in controllers]
        controllers[0].step(FLAGS.step_mul)
        controllers[1].step(FLAGS.step_mul)
        # TODO(zh) combine action in this step with observation several steps earlier due to human reaction time
        while True:
            # 1v1 version
            obs = [controller.observe() for controller in controllers]
            # ISSUE(zh) should we abandon invalid actions
            actions = [o.actions for o in obs]
            if len(actions[0]) > 0 or len(actions[1]) > 0:
                # parse observation
                base_obs = [feat.transform_obs(o) for feat, o in zip(feats, prev_obs)]
                try:
                    agent_obs = [self.obs_parser.parse(o) for o in base_obs]
                except KeyError as e:
                    error_set.add(repr(e).split('_')[-2])
                    if obs[0].player_result:
                        return (step_data, map_size,
                                [{'action_statistics': action_statistics[idx], 'cumulative_statistics':
                                  cumulative_statistics[idx], 'begin_statistics': begin_statistics[idx]}
                                    for idx in range(2)])
                    prev_obs = obs
                    controllers[0].step(FLAGS.step_mul)
                    controllers[1].step(FLAGS.step_mul)
                    print('step', step, error_set)
                    step += FLAGS.step_mul
                    delay[0] += FLAGS.step_mul
                    delay[1] += FLAGS.step_mul
                    continue

                # add obs from the enemy obs
                agent_obs[0]['scalar_info']['enemy_upgrades'] = agent_obs[1]['scalar_info']['upgrades']
                agent_obs[1]['scalar_info']['enemy_upgrades'] = agent_obs[0]['scalar_info']['upgrades']

            for idx in range(N):
                # non-empty action validate
                if len(actions[idx]) == 0:
                    continue
                # merge action info into obs
                result_obs = self.obs_parser.merge_action(agent_obs[idx], last_actions[idx])

                # compress obs
                compressed_obs = compress_obs(result_obs)

                # save statistics and frame(all the action in actions use the same obs)
                for action in actions[idx]:
                    act_raw = action.action_raw
                    agent_acts = act_parser.parse(act_raw)
                    for i, (_, v) in enumerate(agent_acts.items()):
                        v['delay'] = torch.LongTensor([delay[idx]])
                        valid = update_action_stat(action_statistics[idx], v, base_obs[idx])
                        if valid:
                            update_cum_stat(cumulative_statistics[idx], v)
                            if len(begin_statistics[idx]) < begin_num:
                                update_begin_stat(begin_statistics[idx], v)
                            compressed_obs.update({'actions': v})
                            # torch.save(compressed_obs, os.path.join(self.output_dir, '{}.pt'.format(action_count)))
                            step_data[idx].append(compressed_obs)
                            action_count[idx] += 1
                        last_actions[idx] = v
                delay[idx] = 0

            if obs[0].player_result:
                return (step_data, map_size,
                        [{'action_statistics': action_statistics[idx], 'cumulative_statistics':
                          cumulative_statistics[idx], 'begin_statistics': begin_statistics[idx]} for idx in range(2)])

            prev_obs = obs
            controllers[0].step(FLAGS.step_mul)
            controllers[1].step(FLAGS.step_mul)
            if step % 1000 == 0:
                print('step', step)
            step += FLAGS.step_mul
            delay[0] += FLAGS.step_mul
            delay[1] += FLAGS.step_mul


def main(unused_argv):
    run_config = run_configs.get()
    replay_list = sorted(run_config.replay_paths(FLAGS.replays))
    version = replay.get_replay_version(run_config.replay_data(replay_list[0]))
    run_config = run_configs.get(version=version)

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist.".format(FLAGS.replays))

    try:
        p = ReplayProcessor(run_config, output_dir=FLAGS.output_dir)
        p.run(FLAGS.replays)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")


def replay_decode(paths, version):
    '''
        Overview: decode replays and gather results
        Argumens: 
            - paths (:obj:'string'): replays directory
            - version (:obj:'Version'): game version 
        Returns:
            - success_msg (:obj'list'): a list of process success message
            - error_msg (:obj'list'): a list of process error message
    '''        
    run_config = run_configs.get(version=version)
    p = ReplayProcessor(run_config, output_dir=FLAGS.output_dir)
    success_msg = []
    error_msg = []
    for idx, path in enumerate(paths):
        print('idx', idx, path)
        try:
            ret = p.run(path)
            print('{}/{}---{}'.format(idx, len(paths), ret))
            if "invalid" in ret:
                error_msg.append(ret)
            else:
                success_msg.append(ret)
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting.")
        except Exception as e:
            print(e)
            error_msg.append(repr(e))
    return success_msg, error_msg


def main_multi(unused_argv):
    from multiprocessing import Pool
    run_config = run_configs.get()
    replay_list = sorted(run_config.replay_paths(FLAGS.replays))
    version = replay.get_replay_version(run_config.replay_data(replay_list[0]))

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist.".format(FLAGS.replays))

    def log_func(s, idx):
        return s + '\n{}\t'.format(idx) + '-'*60 + '\n'

    def combine_msg(msg):
        msg = list(zip(*msg))
        success_msg, error_msg = msg
        new_success_msg, new_error_msg = [], []
        for item in success_msg:
            new_success_msg += item
        for item in error_msg:
            new_error_msg += item
        new_success_msg = [log_func(s, idx) for idx, s in enumerate(new_success_msg)]
        new_error_msg = [log_func(s, idx) for idx, s in enumerate(new_error_msg)]
        return new_success_msg, new_error_msg

    N = 20
    pool = Pool(N)
    group_num = int(len(replay_list) // N)
    print('total len: {}, group: {}, each group: {}'.format(len(replay_list), N, group_num))
    # ISSUE(zh) splited group number doesn't match pool
    replay_split_list = [replay_list[i*group_num:(i+1)*group_num] for i in range(group_num)]
    func = partial(replay_decode, version=version)
    ret = pool.map(func, replay_split_list)
    success_msg, error_msg = combine_msg(ret)
    with open(os.path.join(FLAGS.output_dir, 'success.txt'), 'w') as f:
        f.writelines(success_msg)
    with open(os.path.join(FLAGS.output_dir, 'error.txt'), 'w') as f:
        f.writelines(error_msg)
    pool.close()
    pool.join()


if __name__ == "__main__":
    app.run(main)
