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
"""Play as a human against an agent by setting up a LAN game.

This needs to be called twice, once for the server, and once for the client.

The server(agent1) plays on the host. There you run it as:
$ python -m pysc2.bin.agent_vs_agent --server --map AbyssalReef --agent tstarbot.agents.zerg_agent.ZergAgent --agent_race zerg

And on the machine the client(agent2) plays on:
$ python -m pysc2.bin.agent_vs_agent --noserver --agent tstarbot.agents.zerg_agent.ZergAgent --agent_race zerg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import logging
import sys
import time

from absl import app
from absl import flags
import portpicker

from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import lan_sc2_env
#from pysc2.env import run_loop
from pysc2.env import sc2_env, lan_server_sc2_env
from pysc2.lib import point_flag

from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,
                  # pylint: disable=protected-access
                  "Agent's race.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", "256",
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", "128",
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", "FEATURES",
                  sc2_env.ActionSpace._member_names_,
                  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")

flags.DEFINE_string("host", "127.0.0.1", "Game Host. Can be 127.0.0.1 or ::1")
flags.DEFINE_integer("port0", 14380, "port0")
flags.DEFINE_integer("port1", 14381, "port1")
flags.DEFINE_integer("port2", 14382, "port2")
flags.DEFINE_integer("port3", 14383, "port3")
flags.DEFINE_integer("port4", 14384, "port4")

flags.DEFINE_string("map", None, "Name of a map to use to play.")
flags.DEFINE_bool("server", True, "Is server or client")
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_integer("max_steps", 0, "max game steps, 0 means no limit")
flags.DEFINE_bool("disable_fog", False, "Whether to disable fog of war or not")
flags.DEFINE_string("agent_config", "",
                    "Agent's config in py file. Pass it as python module."
                    "E.g., tstarbot.agents.dft_config")

def main(unused_argv):
    if FLAGS.server:
        server()
    else:
        client()


def server():
    """Run a host which expects one player to connect remotely."""
    run_config = run_configs.get()

    map_inst = maps.get(FLAGS.map)

    if not FLAGS.rgb_screen_size or not FLAGS.rgb_minimap_size:
        logging.info(
            "Use --rgb_screen_size and --rgb_minimap_size if you want rgb "
            "observations.")

    ports = [FLAGS.port0, FLAGS.port1, FLAGS.port2, FLAGS.port3, FLAGS.port4]
    if not all(portpicker.is_port_free(p) for p in ports):
        sys.exit("Need 5 free ports after the config port.")

    proc = None
    tcp_conn = None

    try:
        proc = run_config.start(extra_ports=ports[1:], timeout_seconds=300,
                                host=FLAGS.host, window_loc=(50, 50))

        tcp_port = ports[0]
        settings = {
            "remote": False,
            "game_version": proc.version.game_version,
            "realtime": FLAGS.realtime,
            "map_name": map_inst.name,
            "map_path": map_inst.path,
            "map_data": map_inst.data(run_config),
            "ports": {
                "server": {"game": ports[1], "base": ports[2]},
                "client": {"game": ports[3], "base": ports[4]},
            }
        }

        create = sc_pb.RequestCreateGame(
            realtime=settings["realtime"],
            local_map=sc_pb.LocalMap(map_path=settings["map_path"]),
            disable_fog=FLAGS.disable_fog)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Participant)

        controller = proc.controller
        controller.save_map(settings["map_path"], settings["map_data"])
        controller.create_game(create)

        print("-" * 80)
        print("Join: agent_vs_agent --host %s --config_port %s" % (proc.host,
                                                                   tcp_port))
        print("-" * 80)

        tcp_conn = lan_sc2_env.tcp_server(
            lan_sc2_env.Addr(proc.host, tcp_port), settings)

        join = sc_pb.RequestJoinGame()
        join.shared_port = 0  # unused
        join.server_ports.game_port = settings["ports"]["server"]["game"]
        join.server_ports.base_port = settings["ports"]["server"]["base"]
        join.client_ports.add(game_port=settings["ports"]["client"]["game"],
                              base_port=settings["ports"]["client"]["base"])

        join.race = sc2_env.Race[FLAGS.agent_race]
        join.options.raw = True
        join.options.score = True
        if FLAGS.feature_screen_size and FLAGS.feature_minimap_size:
            fl = join.options.feature_layer
            fl.width = 24
            FLAGS.feature_screen_size.assign_to(fl.resolution)
            FLAGS.feature_minimap_size.assign_to(fl.minimap_resolution)

        if FLAGS.rgb_screen_size and FLAGS.rgb_minimap_size:
            FLAGS.rgb_screen_size.assign_to(join.options.render.resolution)
            FLAGS.rgb_minimap_size.assign_to(
                join.options.render.minimap_resolution)

        controller.join_game(join)

        with lan_server_sc2_env.LanServerSC2Env(
                race=sc2_env.Race[FLAGS.agent_race],
                step_mul=FLAGS.step_mul,
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen=FLAGS.feature_screen_size,
                    feature_minimap=FLAGS.feature_minimap_size,
                    rgb_screen=FLAGS.rgb_screen_size,
                    rgb_minimap=FLAGS.rgb_minimap_size,
                    action_space=FLAGS.action_space,
                    use_feature_units=FLAGS.use_feature_units),
                visualize=False,
                controller=controller,
                map_name=FLAGS.map) as env:
            agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
            agent_cls = getattr(importlib.import_module(agent_module),
                                agent_name)
            agent_kwargs = {}
            if FLAGS.agent_config:
                agent_kwargs['config_path'] = FLAGS.agent_config
            agents = [agent_cls(**agent_kwargs)]

            try:
                run_loop(agents, env, FLAGS.max_steps)
            except lan_server_sc2_env.RestartException:
                pass

            if FLAGS.save_replay:
                env.save_replay(agent_cls.__name__)
    finally:
        if tcp_conn:
            tcp_conn.close()
        if proc:
            proc.close()


def client():
    """Run the agent, connecting to a (remote) host started independently."""
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    logging.info("Starting agent:")
    with lan_sc2_env.LanSC2Env(
            host=FLAGS.host,
            config_port=FLAGS.port0,
            race=sc2_env.Race[FLAGS.agent_race],
            step_mul=FLAGS.step_mul,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                action_space=FLAGS.action_space,
                use_feature_units=FLAGS.use_feature_units),
            visualize=False) as env:
        agent_kwargs = {}
        if FLAGS.agent_config:
            agent_kwargs['config_path'] = FLAGS.agent_config
        agents = [agent_cls(**agent_kwargs)]
        logging.info("Connected, starting run_loop.")
        try:
            run_loop(agents, env, FLAGS.max_steps)
        except lan_sc2_env.RestartException:
            pass

        if FLAGS.save_replay:
            env.save_replay(agent_cls.__name__)
    logging.info("Done.")


def run_loop(agents, env, max_steps=0):
    """A run loop to have agents and an environment interact."""
    me_id = 0
    total_frames = 0
    n_win = 0
    outcome = 0
    start_time = time.time()

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    for agent in agents:
        agent.setup(observation_spec, action_spec)

    try:
        timesteps = env.reset()
        for a in agents:
            a.reset()

        # run this episode
        while True:
            total_frames += 1
            actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
            timesteps = env.step(actions)
            if timesteps[me_id].last():
                break

            if max_steps and total_frames >= max_steps:
                break

        outcome = timesteps[me_id].reward

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))

    # print info
    if outcome > 0:
        n_win += 1
    elif outcome == 0:
        n_win += 0.5

    print('episode = {}, outcome = {}, n_win = {}, current winning rate = {}'.format(
        0, outcome, n_win, n_win))

def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
