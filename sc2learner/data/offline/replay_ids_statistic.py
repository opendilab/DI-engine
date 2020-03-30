# coding: utf-8
'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Replays ids statistics, including abilities, units, upgrades, buffs, effects
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import time
import sys
import queue
import traceback
from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import replay
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib import actions

VERSION = '4.10.0'

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "/mnt/lustre/zhouhang2/bin/replay_list.txt", "File includes replay path")
flags.DEFINE_string("output_dir", "/mnt/lustre/zhouhang2/data/sc2_lib", "Path to save data")
flags.DEFINE_string("version", "4.10.0", "Game version")
flags.DEFINE_integer("process_num", None, "how many process to run")
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("output_dir")
flags.mark_flag_as_required("process_num")
flags.FLAGS(sys.argv)


class ReplayDecoder(multiprocessing.Process):
    def __init__(self, run_config, replay_queue, output_dir, result, proc_id):
        super(ReplayDecoder, self).__init__()
        self.run_config = run_config
        self.output_dir = output_dir
        self.result = result
        self.replay_queue = replay_queue
        self.pro_id = proc_id
        self.interface = sc_pb.InterfaceOptions(
            raw=True, score=False,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        self.units = set()
        self.upgrades = set()
        self.abilities = set()
        self.effects = set()
        self.buffs = set()

    def _safe_start_game(self):
        flag = 1
        while flag:
            try:
                sc_process = self.run_config.start(want_rgb=False)
                controller = sc_process.controller
                flag = 0
            except:
                self._print('sc2 start failed')
                pass
        return sc_process, controller

    def replay_decode(self, controller, replay_path, game_loops):
        for i in [1, 2]:
            controller.start_replay(sc_pb.RequestStartReplay(
                replay_path=replay_path,
                options=self.interface,
                observed_player_id= i))
            data = controller.data_raw()
            if (len(data.abilities) != 3801 or len(data.units) != 1970 or len(data.upgrades) != 296 or
                    len(data.buffs) != 290 or len(data.effects) != 13):
                raise Exception('stableid does not match')
            cur_loop = 0
            while cur_loop < game_loops:
                next_loop = min(game_loops, cur_loop + 1000)
                try:
                    controller.step(next_loop - cur_loop)
                    ob = controller.observe()
                    for i in ob.actions:
                        if i.HasField('action_raw'):
                            ac = i.action_raw
                            if ac.HasField('unit_command'):
                                self.abilities.add(ac.unit_command.ability_id)
                            if ac.HasField('toggle_autocast'):
                                self.abilities.add(ac.toggle_autocast.ability_id)
                    ob = ob.observation.raw_data
                    if ob.HasField('player'):
                        for i in ob.player.upgrade_ids:
                            self.upgrades.add(i)
                    for i in ob.units:
                        for j in i.buff_ids:
                            self.buffs.add(j)
                        self.units.add(i.unit_type)
                    for i in ob.effects:
                        self.effects.add(i.effect_id)
                except:
                    if next_loop == game_loops:
                        pass
                    else:
                        raise Exception('decoding went wrong')
                cur_loop = next_loop

    def run(self):
        self._print('start SC2')
        sc_process, controller = self._safe_start_game()
        self._print('SC2 start successfully')
        while True:
            try:
                replay_path = self.replay_queue.get(block=True, timeout=1)
            except queue.Empty:
                self._print('done')
                return
            try:
                replay_data = self.run_config.replay_data(replay_path)
                info = controller.replay_info(replay_data)
                self.replay_decode(controller, replay_path, info.game_duration_loops)
                self.result.put([self.abilities, self.units, self.upgrades, self.buffs, self.effects])
                self._print('success paring: ' + replay_path)
            except Exception as e:
                self._print(repr(e) + '\t' + replay_path + '\t' + 'restarting sc2')
                sc_process.close()
                sc_process, controller = self._safe_start_game()

    def _print(self, string):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('[{}] {}: {}'.format(t, self.pro_id, string))


def filter_replay(q, nq):
    run_config = run_configs.get()
    while True:
        try:
            path = q.get(block=True, timeout=1)
        except queue.Empty:
            return
        version = replay.get_replay_version(run_config.replay_data(path.strip()))
        if version.game_version == VERSION:
            nq.put(path.strip())


def result_print(result_queue):
    abilities = set()
    units = set()
    upgrades = set()
    buffs = set()
    effects = set()
    number = 0
    while True:
        number += 1
        unknown_abilities = []
        result = result_queue.get()
        if result is None:
            return
        else:
            abilities.update(result[0])
            units.update(result[1])
            upgrades.update(result[2])
            buffs.update(result[3])
            effects.update(result[4])
        for i in abilities:
            if i not in actions.RAW_ABILITY_IDS:
                unknown_abilities.append(i)
        with open(os.path.join(FLAGS.output_dir, 'result.txt'), 'w+') as f:
            f.writelines('processed replay number:' + str(number) + '\n')
            f.writelines('abilities: ' + str(sorted(abilities)) + '\n')
            f.writelines('units: ' + str(sorted(units)) + '\n')
            f.writelines('upgrades: ' + str(sorted(upgrades)) + '\n')
            f.writelines('buffs: ' + str(sorted(buffs)) + '\n')
            f.writelines('effects: ' + str(sorted(effects)) + '\n')
            f.writelines('unknown abilites: ' + str(sorted(unknown_abilities)) + '\n')


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)    # mute remote_controller, get clear stdout
    run_config = run_configs.get(FLAGS.version)
    replay_queue = multiprocessing.Manager().Queue()
    new_replay_queue = multiprocessing.Manager().Queue()
    # filter replays by version
    with open(FLAGS.replays, 'r+') as f:
        replay_list = f.readlines()
    for i in replay_list:
        replay_queue.put(i)
    print('original replay number:', replay_queue.qsize())
    p = multiprocessing.Pool(100)
    for i in range(100):
        p.apply_async(filter_replay, args=(replay_queue, new_replay_queue))
    p.close()
    p.join()
    print('fitler by version, replays left number:', new_replay_queue.qsize())
    queue = multiprocessing.Queue()  # multiprocess.Manager().Queue() doesn't work with multiprocess.Process,change it
    while not new_replay_queue.empty():
        queue.put(new_replay_queue.get())
    decoders = []
    result_queue = multiprocessing.Queue()
    result_printer = multiprocessing.Process(target=result_print, args=(result_queue,))
    result_printer.start()
    for i in range(FLAGS.process_num):
        decoder = ReplayDecoder(run_config, queue, FLAGS.output_dir, result_queue, i)
        decoder.start()
        decoders.append(decoder)
    for i in decoders:
        i.join()
    result_queue.put(None)  # tell result printer to shut down
    result_printer.join()


if __name__ == "__main__":
    app.run(main)
