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
from sc2learner.envs.statistics import Statistics, transform_stat
from sc2learner.envs.maps.map_info import LOCALIZED_BNET_NAME_TO_PYSC2_NAME_LUT

logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS
# flags.DEFINE_string("replays", "E:/data/replays", "Path to a directory of replays.")
# flags.DEFINE_string("output_dir", "E:/data/replay_data", "Path to save data")
flags.DEFINE_string("version", "4.10.0", "Game version")
# flags.DEFINE_integer("process_num", 1, "Number of sc2 process to start on the node")
# flags.DEFINE_bool(
#     "check_version", False, "Check required game version of the replays and discard ones not matching version"
# )
# flags.mark_flag_as_required("replays")
# flags.mark_flag_as_required("output_dir")
flags.FLAGS(sys.argv)

def parse_info(info, replay_path):
    if (info.HasField("error")):
        print('{} have error'.format(replay_path))
        return []

    race_dict = {1: 'Terran', 2: 'Zerg', 3: 'Protoss'}
    returns = []
    for home in range(2):
        away = 1 if home == 0 else 0
        ret = dict()
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
        ret['screen_resolution'] = 1  # placeholder
        returns.append(ret)
    return returns


def time_format(time_item):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_item))


def main(unused_argv):
    sc2_version = FLAGS.version
    # sc2_version = '4.8.0'
    # sc2_version = '4.8.2'
    # sc2_version = '4.8.3'
    # sc2_version = '4.8.4'
    # sc2_version = '4.8.6'
    # sc2_version = '4.9.0'
    # sc2_version = '4.9.1'
    # sc2_version = '4.9.2'
    # sc2_version = '4.9.3'
    os.environ['SC2PATH'] = '/mnt/lustre/zhangming/StarCraftII_Linux_Package/StarCraftII_' + sc2_version
    run_config = run_configs.get(sc2_version)
    handle = run_config.start(want_rgb=False)
    controller = handle.controller

    results = {}
    save_dir = '/mnt/lustre/zhangming/data/meta_data'

    replay_list_path = '/mnt/lustre/zhangming/data/list/raw_replay/{}.list'.format(sc2_version)
    replays = open(replay_list_path, 'r').readlines()
    for index, replay_path in enumerate(replays):
        if index % 1 == 0:
            print('{} {}/{}'.format(time_format(time.time()), len(results), index))
        try:
            replay_path = replay_path.strip()
            replay_data = run_config.replay_data(replay_path)
            info = controller.replay_info(replay_data)
            result = parse_info(info, replay_path)
            results[replay_path] = result
        except Exception as e:
            print('[error][{}] {}'.format(replay_path, str(e)))
    torch.save(results, os.path.join(save_dir, sc2_version + '.meta'))


if __name__ == "__main__":
    app.run(main)
