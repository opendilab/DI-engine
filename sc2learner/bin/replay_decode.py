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
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarObsParser
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 1, "How many game steps per observation.")
flags.DEFINE_string("replays", None, "Path to a directory of replays.")
flags.mark_flag_as_required("replays")


size = point.Point(16, 16)
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
        # if p.player_apm < 10 or p.player_mmr < 1000:
        if p.player_apm < 10:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return False
    return True


class ReplayProcessor(multiprocessing.Process):
    """A Process that pulls replays and processes them."""

    def __init__(self, run_config, output_dir=None):
        super(ReplayProcessor, self).__init__()
        self.run_config = run_config
        self.output_dir = output_dir
        self.output_dir = '/mnt/lustre/niuyazhe/code/gitlab/SenseStar/sc2learner/bin/test_data'
        self.obs_parser = AlphastarObsParser()
        self.act_parser = AlphastarActParser()
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
            return replay_data, map_data
        else:
            self._print("Replay is invalid.")
            return None

    def run(self, replay_path):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        self._print("Starting up a new SC2 instance.")
        try:
            ret = self._replay_prepare(self.controllers[0], replay_path)
            if ret is not None:
                replay_data, map_data = ret
                self.process_replay_multi(self.controllers, replay_data, map_data, self.player_ids)
            else:
                return
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
        N = len(player_ids)
        step = 0
        delay = [0 for _ in range(N)]
        action_count = 0
        # delay, queued, action_type, selected_units, target_units
        last_info = [([0], [0], [0], [0], [0]) for _ in range(N)]
        while True:
            # 1v1 version
            obs = [controller.observe() for controller in controllers]
            agent_obs = [feat.transform_obs(o) for feat, o in zip(feats, obs)]
            agent_obs = [self.obs_parser.parse(o) for o in agent_obs]

            agent_obs[0]['scalar_info']['enemy_upgrades'] = agent_obs[1]['scalar_info']['upgrades']
            agent_obs[1]['scalar_info']['enemy_upgrades'] = agent_obs[0]['scalar_info']['upgrades']

            actions = [o.actions for o in obs]
            if len(actions[1]) > 0:
                for action in actions[1]:
                    act_raw = action.action_raw
                    agent_acts = self.act_parser.parse(act_raw)
                    for idx, (_, v) in enumerate(agent_acts.items()):
                        v['delay'] = torch.LongTensor([delay[1]])
                        delay[1] = 0
                        last_info[1] = (v['delay'], v['queued'], v['action_type'],
                                        v['selected_units'], v['target_units'])
            if len(actions[0]) > 0:
                for action in actions[0]:
                    act_raw = action.action_raw
                    agent_acts = self.act_parser.parse(act_raw)
                    for idx, (_, v) in enumerate(agent_acts.items()):
                        v['delay'] = torch.LongTensor([delay[0]])
                        delay[0] = 0
                        last_info[0] = (v['delay'], v['queued'], v['action_type'],
                                        v['selected_units'], v['target_units'])
                        agent_obs[0] = self.obs_parser.merge_action(agent_obs[0], last_info[0])
                        agent_obs[1] = self.obs_parser.merge_action(agent_obs[1], last_info[1])
                        print(v)
                        torch.save(
                            {'obs0': agent_obs[0], 'obs1': agent_obs[1], 'act': v},
                            os.path.join(self.output_dir, '{}.pt'.format(action_count))
                        )
                        print('save in {}'.format(os.path.join(self.output_dir, '{}.pt'.format(action_count))))
                        action_count += 1

            if obs[0].player_result:
                return

            controllers[0].step(FLAGS.step_mul)
            controllers[1].step(FLAGS.step_mul)
            print('step', step)
            step += FLAGS.step_mul
            delay[0] += FLAGS.step_mul
            delay[1] += FLAGS.step_mul


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
    run_config = run_configs.get()
    replay_list = sorted(run_config.replay_paths(FLAGS.replays))
    version = replay.get_replay_version(run_config.replay_data(replay_list[0]))
    run_config = run_configs.get(version=version)

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist.".format(FLAGS.replays))

    try:
        p = ReplayProcessor(run_config)
        p.run(FLAGS.replays)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")


if __name__ == "__main__":
    app.run(main)
