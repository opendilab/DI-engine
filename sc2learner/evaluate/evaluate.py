from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import yaml
from easydict import EasyDict
import random
import torch
from multiprocessing import Pool
from functools import partial
import math

from absl import app
from absl import flags
from absl import logging

from sc2learner.envs.alphastar_env import AlphaStarEnv
from sc2learner.worker import RandomAgent, AlphaStarAgent
from sc2learner.utils import build_logger
from pysc2.lib.action_dict import ACTION_INFO_MASK

FLAGS = flags.FLAGS
flags.DEFINE_string("job_name", "", "actor or learner")
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("replay_path", "", "folder name in /StarCraftII/Replays to save the evaluate replays")
flags.DEFINE_integer("difficulty", 1, "difficulty of bot to play with")
flags.FLAGS(sys.argv)


def create_env(cfg, random_seed=None):
    cfg.env.random_seed = random_seed
    env = AlphaStarEnv(cfg)
    return env


def evaluate(var_dict, cfg):

    game_seed, rank = var_dict['game_seed'], var_dict['rank']
    log_time = time.strftime("%d:%m:%H:%M:%S")
    if cfg.common.load_path != "":
        name_list = cfg.common.load_path.split('/')
        path_info = name_list[-2] + '/' + name_list[-1].split('.')[0]
    else:
        path_info = 'no_model'
    name = 'eval_{}_{}_{}_{}'.format(path_info, cfg.env.difficulty, log_time, rank + 1)
    dirname = os.path.dirname(name)
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            pass
    logger, tb_logger, _ = build_logger(cfg, name=name)
    logger.info('cfg: {}'.format(cfg))
    logger.info("Rank %d Game Seed: %d" % (rank, game_seed))
    env = create_env(cfg, game_seed)

    if cfg.common.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif cfg.common.agent == 'alphastar':
        agent = AlphaStarAgent(cfg)
    else:
        raise NotImplementedError

    cum_return = 0.0
    action_counts = [0] * (max(ACTION_INFO_MASK.keys()) + 1)

    observation = env.reset()
    agent.reset()
    done, step_id = False, 0
    while not done:
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        logger.info("Rank %d Step ID: %d Take Action: %s" % (rank, step_id, env.cur_actions))
        action_counts[env.cur_action_type] += 1
        cum_return += reward
        step_id += 1
    if cfg.env.save_replay:
        env.save_replay(cfg.common.agent + FLAGS.replay_path)
    for action_id in ACTION_INFO_MASK.keys():
        logger.info(
            "Rank %d\tAction ID: %d\tCount: %d\tName: %s" %
            (rank, action_id, action_counts[action_id], ACTION_INFO_MASK[action_id]['name'])
        )
    env.close()
    return cum_return


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.env.difficulty = FLAGS.difficulty

    base_dir = os.environ.get("SC2PATH", "~/StarCraftII")
    base_dir = os.path.expanduser(base_dir)
    if not os.path.exists(base_dir + "/Replays/" + cfg.common.agent + FLAGS.replay_path):
        os.mkdir(base_dir + "/Replays/" + cfg.common.agent + FLAGS.replay_path)
    use_multiprocessing = cfg.common.get("use_multiprocessing", True)
    if use_multiprocessing:
        eval_func = partial(evaluate, cfg=cfg)
        pool = Pool(min(cfg.common.num_episodes, 20))
        var_list = []
        for i in range(cfg.common.num_episodes):
            seed = random.randint(0, math.pow(2, 32) - 1)
            var_list.append({'rank': i, 'game_seed': seed})

        reward_list = pool.map(eval_func, var_list)
        print(reward_list)
        pool.close()
    else:
        reward_list = []
        for i in range(cfg.common.num_episodes):
            seed = random.randint(0, math.pow(2, 32) - 1)
            reward = evaluate({'rank': 0, 'game_seed': seed}, cfg)
            reward_list.append(reward)

    print(
        "Evaluated %d Episodes Against Bot Level %s Avg Return %f Avg Winning Rate %f" % (
            cfg.common.num_episodes, cfg.env.difficulty, sum(reward_list) / len(reward_list),
            ((sum(reward_list) / len(reward_list)) + 1) / 2.0
        )
    )


if __name__ == '__main__':
    app.run(main)
