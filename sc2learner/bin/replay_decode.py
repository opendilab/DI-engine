#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dump out stats about all the actions that are in use in a set of replays."""

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
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser
from functools import partial

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 1, "How many game steps per observation.")
flags.DEFINE_string("replays", None, "Path to a directory of replays.")
flags.DEFINE_string("output_dir", "/mnt/lustre/niuyazhe/data/sl_data_new", "Path to save data")
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
    """Make sure the replay isn't corrupt, and is worth looking at."""
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
    """A Process that pulls replays and processes them."""

    def __init__(self, run_config, output_dir=None):
        super(ReplayProcessor, self).__init__()
        assert(output_dir is not None)
        self.run_config = run_config
        self.output_dir = output_dir
        self.obs_parser = AlphastarObsParser()
        self.handles = []
        self.controllers = []
        self.player_ids = [i+1 for i in range(2)]
        for i in self.player_ids:
            handle = self.run_config.start(want_rgb=interface.HasField("render"))
            controller = handle.controller
            self.handles.append(handle)
            self.controllers.append(controller)
        self._print("SC2 Started successfully.")

    def _replay_prepare(self, controller, replay_path, print_info=True):
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
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        self._print("Starting up a new SC2 instance.")
        try:
            ret = self._replay_prepare(self.controllers[0], replay_path)
            if ret is not None:
                replay_data, map_data, info = ret
                meta_data_0 = self._parse_info(info, replay_path, home=0)
                meta_data_1 = self._parse_info(info, replay_path, home=1)
                step_data_0, stat0, map_size = self.process_replay_multi(
                    self.controllers, replay_data, map_data, self.player_ids)
                step_data_1, stat1, map_size = self.process_replay_multi(
                    self.controllers, replay_data, map_data, list(reversed(self.player_ids)))
                meta_data_0['step_num'] = len(step_data_0)
                meta_data_1['step_num'] = len(step_data_1)
                meta_data_0['map_size'] = map_size
                meta_data_1['map_size'] = map_size
                name0 = '{}_{}_{}_{}'.format(meta_data_0['home_race'], meta_data_0['away_race'],
                                             meta_data_0['home_mmr'], os.path.basename(replay_path).split('.')[0])
                name1 = '{}_{}_{}_{}'.format(meta_data_1['home_race'], meta_data_1['away_race'],
                                             meta_data_1['home_mmr'], os.path.basename(replay_path).split('.')[0])
                torch.save(meta_data_0, os.path.join(self.output_dir, name0+'.meta'))
                torch.save(step_data_0, os.path.join(self.output_dir, name0+'.step'))
                torch.save(stat0, os.path.join(self.output_dir, name0+'.stat'))
                torch.save(meta_data_1, os.path.join(self.output_dir, name1+'.meta'))
                torch.save(step_data_1, os.path.join(self.output_dir, name1+'.step'))
                torch.save(stat1, os.path.join(self.output_dir, name1+'.stat'))
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
        feats = []
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
        act_parser = AlphastarActParser(feature_layer_resolution=RESOLUTION, map_size=map_size)
        N = len(player_ids)
        step = 0
        delay = [0 for _ in range(N)]
        action_count = 0
        # delay, queued, action_type, selected_units, target_units
        last_info = [([0], [0], [0], 'none', 'none') for _ in range(N)]

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
                        continue
                    action_statistics[action_type]['selected_type'].add(unit_type)
            if isinstance(act['target_units'], torch.Tensor):
                for unit_tag in act['target_units']:
                    unit_type = get_unit_type(unit_tag.item(), obs)
                    if unit_type is None:
                        print("not found target unit(id: {})".format(unit_tag.item()))
                        continue
                    action_statistics[action_type]['target_type'].add(unit_type)

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

        step_data = []
        error_set = set()
        action_statistics = {}
        cumulative_statistics = {}
        begin_statistics = []
        begin_num = 100

        while True:
            # 1v1 version
            obs = [controller.observe() for controller in controllers]
            base_obs = [feat.transform_obs(o) for feat, o in zip(feats, obs)]
            try:
                agent_obs = [self.obs_parser.parse(o) for o in base_obs]
            except KeyError as e:
                error_set.add(repr(e).split('_')[-2])
                if obs[0].player_result:
                    return step_data, {'action_statistics': action_statistics, 'cumulative_statistics':
                                       cumulative_statistics, 'begin_statistics': begin_statistics}, map_size
                controllers[0].step(FLAGS.step_mul)
                controllers[1].step(FLAGS.step_mul)
                print('step', step, error_set)
                step += FLAGS.step_mul
                continue

            agent_obs[0]['scalar_info']['enemy_upgrades'] = agent_obs[1]['scalar_info']['upgrades']
            agent_obs[1]['scalar_info']['enemy_upgrades'] = agent_obs[0]['scalar_info']['upgrades']

            actions = [o.actions for o in obs]
            if len(actions[1]) > 0:
                for action in actions[1]:
                    act_raw = action.action_raw
                    agent_acts = act_parser.parse(act_raw)
                    for idx, (_, v) in enumerate(agent_acts.items()):
                        v['delay'] = torch.LongTensor([delay[1]])
                        delay[1] = 0
                        last_info[1] = (v['delay'], v['queued'], v['action_type'],
                                        v['selected_units'], v['target_units'])
            if len(actions[0]) > 0:
                for action in actions[0]:
                    act_raw = action.action_raw
                    agent_acts = act_parser.parse(act_raw)
                    for idx, (_, v) in enumerate(agent_acts.items()):
                        v['delay'] = torch.LongTensor([delay[0]])
                        update_action_stat(action_statistics, v, base_obs[0])
                        update_cum_stat(cumulative_statistics, v)
                        if len(begin_statistics) < begin_num:
                            update_begin_stat(begin_statistics, v)
                            print(begin_statistics)
                        delay[0] = 0
                        agent_obs[0] = self.obs_parser.merge_action(agent_obs[0], last_info[0])
                        agent_obs[1] = self.obs_parser.merge_action(agent_obs[1], last_info[1])
                        last_info[0] = (v['delay'], v['queued'], v['action_type'],
                                        v['selected_units'], v['target_units'])
                        # torch.save(
                        #     {'obs0': agent_obs[0], 'obs1': agent_obs[1], 'act': v},
                        #     os.path.join(self.output_dir, '{}.pt'.format(action_count))
                        # )
                        step_data.append({'obs0': agent_obs[0], 'obs1': agent_obs[1], 'act': v})
                        action_count += 1

            if obs[0].player_result:
                return step_data, {'action_statistics': action_statistics, 'cumulative_statistics':
                                   cumulative_statistics, 'begin_statistics': begin_statistics}, map_size

            controllers[0].step(FLAGS.step_mul)
            controllers[1].step(FLAGS.step_mul)
            if step % 1000 == 0:
                print('step', step)
            step += FLAGS.step_mul
            delay[0] += FLAGS.step_mul
            delay[1] += FLAGS.step_mul


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
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
            error_msg.append(repr(e))
    return success_msg, error_msg


def main_multi(unused_argv):
    from multiprocessing import Pool
    """Dump stats about all the actions that are in use in a set of replays."""
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

    N = 40
    pool = Pool(N)
    group_num = int(len(replay_list) // N)
    print('total len: {}, group: {}, each group: {}'.format(len(replay_list), N, group_num))
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
    app.run(main_multi)
