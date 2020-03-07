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
import time
import os
import signal
import sys
from copy import deepcopy

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import torch
from collections import deque

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from pysc2.lib import replay

from pysc2.lib import gfile
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser, compress_obs, decompress_obs, \
    transform_cum_stat, transform_stat
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser, remove_repeat_data
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
        if p.player_apm < 10 or p.player_mmr < 3500:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return False
    if (info.player_info[0].player_info.race_actual != 2 and
            info.player_info[1].player_info.race_actual != 2):
        # not include Zerg race
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
                - run_config (:obj:'RunConfig'): starcraft2 run config
                - output_dir (:obj:'string'): path to save data
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
        self.test_controller = self.run_config.start(want_rgb=interface.HasField("render")).controller

    def _replay_prepare(self, controller, replay_path, print_info=True):
        '''
            Overview: get basic information from a replay and validate it
            Arguments:
                - controller (:obj:'RemoteController'): game controller whick takes actions
                and generates observations in proto format
                - replay_path (:obj:'string'): path to the replay
                - print_info (:obj:'bool'): a bool decides whether to print replay info
            Returns:
                - replay_data (:obj'bytes'): replay file
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
            return replay_data, info
        else:
            self._print("Replay is invalid.")
            return None

    def _parse_info(self, info, replay_path, map_size, home=0):
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
        ret['screen_resolution'] = RESOLUTION
        ret['map_size'] = map_size
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
                # get replay data
                replay_data, info = ret
                # get map_size
                self.test_controller.start_replay(sc_pb.RequestStartReplay(
                    replay_data=replay_data,
                    options=interface,
                    observed_player_id=0))
                map_size = self.test_controller.game_info().start_raw.map_size
                map_size = [map_size.x, map_size.y]
                self.test_controller.quit()
                # prepare meta data
                meta_data_0 = self._parse_info(info, replay_path, map_size, home=0)
                meta_data_1 = self._parse_info(info, replay_path, map_size, home=1)
                # start replay and record step data
                step_data, stat = self.process_replay_multi(
                    self.controllers, replay_data, map_size, self.player_ids)
                # remove repeat data
                step_data = [remove_repeat_data(d) for d in step_data]
                meta_data_0['step_num'] = len(step_data[0])
                meta_data_1['step_num'] = len(step_data[1])
                # transform stat
                stat_processed_0 = transform_stat(stat[0], meta_data_0)
                stat_processed_1 = transform_stat(stat[1], meta_data_1)
                # save data
                name0 = '{}_{}_{}_{}'.format(meta_data_0['home_race'], meta_data_0['away_race'],
                                             meta_data_0['home_mmr'], os.path.basename(replay_path).split('.')[0])
                name1 = '{}_{}_{}_{}'.format(meta_data_1['home_race'], meta_data_1['away_race'],
                                             meta_data_1['home_mmr'], os.path.basename(replay_path).split('.')[0])
                torch.save(meta_data_0, os.path.join(self.output_dir, name0+'.meta'))
                torch.save(step_data[0], os.path.join(self.output_dir, name0+'.step'))
                torch.save(stat[0], os.path.join(self.output_dir, name0+'.stat'))
                torch.save(stat_processed_0, os.path.join(self.output_dir, name0+'.stat_processed'))
                torch.save(meta_data_1, os.path.join(self.output_dir, name1+'.meta'))
                torch.save(step_data[1], os.path.join(self.output_dir, name1+'.step'))
                torch.save(stat[1], os.path.join(self.output_dir, name1+'.stat'))
                torch.save(stat_processed_1, os.path.join(self.output_dir, name1+'.stat_processed'))
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

    def process_replay_multi(self, controllers, replay_data, map_size, player_ids):
        '''
            Overview: decode a replay step by step to generate data
            Arguments:
                - controllers (:obj:'RemoteController'): game controller whick takes actions
                and generates observations in proto format
                - replay_data (:obj:'bytes'): replay file
                - map_size (:obj'list'): map size in [x, y] format
                - player_ids (:obj:'int'): player id to identify player
            Returns:
                - step_data (:obj'list'): step data for both players, includes observations and actions
                - stat (:obj'list'): statistics in the replay for both players
        '''
        feats = []
        map_size_point = point.Point(map_size[0], map_size[1])
        map_size_point.assign_to(interface.feature_layer.minimap_resolution)
        # start replay with specific settings in both controllers
        for controller, player_id in zip(controllers, player_ids):
            controller.start_replay(sc_pb.RequestStartReplay(
                replay_data=replay_data,
                options=interface,
                observed_player_id=player_id))
            # initial features from game info
            feat = features.features_from_game_info(controller.game_info())
            feats.append(feat)

            controller.step()
        # initial an act_parser to parse actions
        act_parser = AlphastarActParser(feature_layer_resolution=RESOLUTION, map_size=map_size)

        def update_action_stat(action_statistics, act, obs):
            def get_unit_types(units, entity_type_dict):
                unit_types = set()
                for u in units:
                    try:
                        unit_type = entity_type_dict[u]
                        unit_types.add(unit_type)
                    except KeyError:
                        print("not found unit(id: {})".format(u))
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
                action_statistics[action_type]['selected_type'] = action_statistics[action_type]['selected_type'].union(unit_types)  # noqa
            if isinstance(act['target_units'], torch.Tensor):
                units = act['target_units'].tolist()
                unit_types = get_unit_types(units, entity_type_dict)
                action_statistics[action_type]['target_type'] = action_statistics[action_type]['target_type'].union(unit_types)  # noqa

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
            for i in range(raw_units.shape[0]):
                if raw_units[i, key_index] == 1879:
                    raw_units[i, key_index] = 1904
                elif raw_units[i, key_index] == 1883:
                    raw_units[i, key_index] = 1908
            return obs

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
        last_step_data = [None for _ in range(N)]
        begin_num = 200
        prev_obs_queue = deque(maxlen=8)  # (len, 2)

        def unit_record(obs, s):
            for idx, ob in enumerate(obs):
                raw_set = ob['entity_raw']['id']
                units = [4352376833]  # for debug
                for u in units:
                    if u in raw_set:
                        i = raw_set.index(u)
                        print(u, idx, ob['entity_raw']['type'][i], s)
        while True:
            # 1v1 version
            obs = [controller.observe() for controller in controllers]
            obs_actions = [o.actions for o in obs]
            # get raw action
            actions = []
            for a in obs_actions:
                if len(a) > 0:
                    actions.append([t.action_raw for t in a])
                else:
                    actions.append([])
            # when one of the player has action, transfrom all the previous observation
            if len(actions[0]) > 0 or len(actions[1]) > 0:
                for i in range(len(prev_obs_queue)):
                    if not isinstance(prev_obs_queue[i][0], dict):  # whether is processed
                        # parse observation
                        base_obs = [feat.transform_obs_fast(o) for feat, o in zip(feats, prev_obs_queue[i])]
                        base_obs = [unit_id_mapping(o) for o in base_obs]
                        agent_obs = [self.obs_parser.parse(o) for o in base_obs]

                        # add obs from the enemy obs
                        agent_obs[0]['scalar_info']['enemy_upgrades'] = agent_obs[1]['scalar_info']['upgrades']
                        agent_obs[1]['scalar_info']['enemy_upgrades'] = agent_obs[0]['scalar_info']['upgrades']
                        prev_obs_queue[i] = agent_obs

            for idx in range(N):
                # parse action
                agent_acts = []
                for action in actions[idx]:
                    agent_acts.extend(act_parser.parse(action))
                # non-empty valid action
                if len(agent_acts) == 0:
                    continue
                # merge action for the same action id
                agent_acts = act_parser.merge_same_id_action(agent_acts)

                # select obs
                agent_obs, agent_acts, obs_idx = self.match_obs_by_action([t[idx] for t in prev_obs_queue],
                                                                          agent_acts, idx, step)

                # save statistics and frame(all the action in actions use the same obs except last action info)
                create_entity_dim = True
                for i, v in enumerate(agent_acts):
                    # add last step action delay
                    if last_step_data[idx] is not None:  # not init step
                        last_step = last_step_data[idx]
                        last_step['actions']['delay'] = torch.LongTensor([delay[idx]])
                        step_data[idx].append(last_step)
                    # update stat
                    update_action_stat(action_statistics[idx], v, agent_obs)
                    update_cum_stat(cumulative_statistics[idx], v)
                    if len(begin_statistics[idx]) < begin_num:
                        update_begin_stat(begin_statistics[idx], v)
                    # merge cumulative_statistics
                    agent_obs['scalar_info']['cumulative_stat'] = transform_cum_stat(cumulative_statistics[idx])
                    # merge action info into obs
                    result_obs = self.obs_parser.merge_action(agent_obs, last_actions[idx], create_entity_dim)
                    result_obs.update({'actions': v})
                    last_step_data[idx] = compress_obs(result_obs)
                    # update info
                    action_count[idx] += 1
                    last_actions[idx] = v
                    delay[idx] = 0
                    create_entity_dim = False

            if obs[0].player_result or obs[1].player_result:
                # add the last action
                for idx in range(N):
                    last_step = last_step_data[idx]
                    last_step['actions']['delay'] = torch.LongTensor([delay[idx]])
                    step_data[idx].append(last_step)
                return (step_data,
                        [{'action_statistics': action_statistics[idx], 'cumulative_statistics':
                          cumulative_statistics[idx], 'begin_statistics': begin_statistics[idx]} for idx in range(2)])

            prev_obs_queue.append(obs)
            controllers[0].step(FLAGS.step_mul)
            controllers[1].step(FLAGS.step_mul)
            if step % 1000 == 0:
                print('step', step)
            step += FLAGS.step_mul
            delay[0] += FLAGS.step_mul
            delay[1] += FLAGS.step_mul

    def match_obs_by_action(self, prev_obs_queue, actions, act_idx, step):
        # units judge
        def units_judge(act, obs):
            def judge(units):
                units_set = set(units)
                obs_units_set = set(obs['entity_raw']['id'])
                return units_set.issubset(obs_units_set)

            def get_mismatch_info(units):
                units_set = set(units)
                obs_units_set = set(obs['entity_raw']['id'])
                return units_set, obs_units_set
            selected_units = act['selected_units']
            target_units = act['target_units']
            if isinstance(selected_units, torch.Tensor):
                units = selected_units.tolist()
                if not judge(units):
                    a, o = get_mismatch_info(units)
                    diff = a-o
                    print('mismatch info({}): {}, {}, {}, {}'.format('selected_units', a, diff, act_idx, step))
                    return False, list(diff)
            if isinstance(target_units, torch.Tensor):
                units = target_units.tolist()
                if not judge(units):
                    a, o = get_mismatch_info(units)
                    diff = a-o
                    print('mismatch info({}): {}, {}, {}, {}'.format('target_units', a, a-o, act_idx, step))
                    return False, None  # diff only in selected_units
            return True, None

        select_obs = {idx - len(prev_obs_queue): t for idx, t in enumerate(prev_obs_queue)}  # negative number key
        legal_act = []
        for act in actions:
            s_units = act['selected_units']
            flag = [False for _ in range(len(select_obs))]
            diff = []
            for idx, (k, obs) in enumerate(select_obs.items()):
                cur_flag, cur_diff = units_judge(act, obs)
                flag[idx] = cur_flag
                diff.append(cur_diff)
            if sum(flag) == 0 and len(s_units) > 1:
                # another selected_units chance(len of selected_units greater than 1)
                miss_units = {}
                one_miss_flag = []
                for d in diff:
                    if d is None or len(d) > 1:
                        one_miss_flag.append(False)
                    else:  # only one element miss
                        one_miss_flag.append(True)
                        for t in d:
                            if t in miss_units.keys():
                                miss_units[t] += 1
                            else:
                                miss_units[t] = 1
                miss_units = [(k, v) for k, v in miss_units.items()]
                if len(miss_units) > 0:  # if has only one element miss case
                    max_miss_units = max(miss_units, key=lambda x: x[1])[0]
                    flag = [f and diff[idx][0] == max_miss_units for idx, f in enumerate(one_miss_flag)]
                if sum(flag) != 0:  # remove the cooresponding selected_units
                    new_s_unit = s_units.tolist()
                    new_s_unit.remove(max_miss_units)
                    act['selected_units'] = torch.tensor(new_s_unit, dtype=s_units.dtype)
                    print('remove selected_units: {}/{}'.format(max_miss_units, act))
            if sum(flag) != 0:
                legal_act.append(act)
                select_obs = {k: t for idx, (k, t) in enumerate(select_obs.items()) if flag[idx]}
            # no matched obs, abandon the action and keep select_obs
            else:
                print('abandon action', act)
                pass  # placeholder
        idx = max(select_obs.keys())
        if idx != -1:
            print('use the non-nearest obs', act_idx, step, idx)
        selected_obs = deepcopy(select_obs[idx])  # the closest obs
        return selected_obs, legal_act, idx


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
