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

import collections
import multiprocessing
import os
import signal
import sys
import threading
import time

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import queue
import six

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from pysc2.lib import replay
from pysc2.lib import static_data

from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2learner.envs.observations.alphastar_obs_wrapper import AlphastarParser

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")
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
        self.obs_parser = AlphastarParser()
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
        count = 0
        while True:
            for idx, (controller, feat) in enumerate(zip(controllers, feats)):
                obs = controller.observe()
                agent_obs = feat.transform_obs(obs)
                self.obs_parser.parse(agent_obs)
                print('controller {} count {}'.format(idx, count))

                if obs.player_result:
                    return

                controller.step(FLAGS.step_mul)
            count += 1

    def process_replay(self, controller, replay_data, map_data, player_id):
        """Process a single replay, updating the stats."""
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        feat = features.features_from_game_info(controller.game_info())

        controller.step()
        while True:
            obs = controller.observe()
            print('obs over')

            for action in obs.actions:
                act_fl = action.action_feature_layer
                if act_fl.HasField("unit_command"):
                    self.stats.replay_stats.made_abilities[
                        act_fl.unit_command.ability_id] += 1
                if act_fl.HasField("camera_move"):
                    self.stats.replay_stats.camera_move += 1
                if act_fl.HasField("unit_selection_point"):
                    self.stats.replay_stats.select_pt += 1
                if act_fl.HasField("unit_selection_rect"):
                    self.stats.replay_stats.select_rect += 1
                if action.action_ui.HasField("control_group"):
                    self.stats.replay_stats.control_group += 1

                try:
                    func = feat.reverse_action(action).function
                except ValueError:
                    func = -1
                self.stats.replay_stats.made_actions[func] += 1

            for valid in obs.observation.abilities:
                self.stats.replay_stats.valid_abilities[valid.ability_id] += 1

            for u in obs.observation.raw_data.units:
                self.stats.replay_stats.unit_ids[u.unit_type] += 1
                for b in u.buff_ids:
                    self.stats.replay_stats.buffs[b] += 1

            for u in obs.observation.raw_data.player.upgrade_ids:
                self.stats.replay_stats.upgrades[u] += 1

            for e in obs.observation.raw_data.effects:
                self.stats.replay_stats.effects[e.effect_id] += 1

            for ability_id in feat.available_actions(obs.observation):
                self.stats.replay_stats.valid_actions[ability_id] += 1

            if obs.player_result:
                break

            controller.step(FLAGS.step_mul)


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
