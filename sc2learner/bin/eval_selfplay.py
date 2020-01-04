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

from sc2learner.envs.selfplay_raw_env import SC2SelfplayRawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergPlayerObservationWrapper
from sc2learner.agents.model import PPOLSTM, PPOMLP
from sc2learner.agents.solver import PpoAgent, RandomAgent, KeyboardAgent
from sc2learner.utils import build_logger

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("agent1_load_path", "", "path to agent1 model checkpoint")
flags.DEFINE_string("agent2_load_path", "", "path to agent2 model checkpoint")
flags.DEFINE_string("replay_path", "", "folder name in /StarCraftII/Replays to save the evaluate replays")
flags.FLAGS(sys.argv)

def create_selfplay_env(cfg, random_seed=None):
    env = SC2SelfplayRawEnv(map_name=cfg.env.map_name,
                            step_mul=cfg.env.step_mul,
                            resolution=cfg.env.resolution,
                            agent_race=cfg.env.agent_race[0],
                            opponent_race=cfg.env.agent_race[1],
                            tie_to_lose=cfg.env.tie_to_lose,
                            disable_fog=cfg.env.disable_fog,
                            game_steps_per_episode=cfg.env.game_steps_per_episode,
                            random_seed=random_seed)
    env = ZergPlayerActionWrapper(
        player=0,
        env=env,
        game_version=cfg.env.game_version,
        mask=cfg.env.use_action_mask,
        use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergPlayerObservationWrapper(
        player=0,
        env=env,
        use_spatial_features=cfg.env.use_spatial_features,
        use_game_progress=(not cfg.model.policy == 'lstm'),
        action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
        use_regions=cfg.env.use_region_features)

    env = ZergPlayerActionWrapper(
        player=1,
        env=env,
        game_version=cfg.env.game_version,
        mask=cfg.env.use_action_mask,
        use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergPlayerObservationWrapper(
        player=1,
        env=env,
        use_spatial_features=cfg.env.use_spatial_features,
        use_game_progress=(not cfg.model.policy == 'lstm'),
        action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
        use_regions=cfg.env.use_region_features)
    print(env.observation_space, env.action_space)
    return env

def create_dqn_agent(cfg, env):
    from sc2learner.agents.dqn_agent import DQNAgent
    from sc2learner.agents.dqn_networks import NonspatialDuelingQNet

    assert cfg.model.policy == 'mlp'
    assert not cfg.env.use_action_mask
    network = NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                                    n_out=env.action_space.n)
    agent = DQNAgent(network, env.action_space, cfg.common.model_path)
    return agent

def create_ppo_agent(cfg, env):
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
                ob_space=env.observation_space,
                ac_space=env.action_space,
                action_type=cfg.model.action_type,
            )
    agent = PpoAgent(env=env, model=model, cfg=cfg)
    return agent

def cfg_process(cfg, idx):
    result = {}
    for k, v in cfg.items():
        if isinstance(v, dict): result[k] = cfg_process(v, idx)
        elif isinstance(v, list): result[k] = v[idx]
        else: result[k] = v
    return EasyDict(result)

def evaluate(var_dict, cfg):
    game_seed, rank = var_dict['game_seed'], var_dict['rank']
    log_time = time.strftime("%d/%m/%H:%M:%S")
    name_list1 = cfg.common.load_path[0].split('/')
    path_info1 = name_list[-2] + '/' + name_list[-1].split('.')[0]
    name_list2 = cfg.common.load_path[1].split('/')
    path_info2 = name_list[-2] + '/' + name_list[-1].split('.')[0]
    name = 'eval_selfplay_{}_{}_{}_{}_{}_{}'.format(path_info1, path_info2, cfg.model.action_type[0],
            cfg.model.action_type[1], log_time, rank+1)
    logger, tb_logger, _ = build_logger(cfg, name=name)
    logger.info('cfg: {}'.format(cfg))
    logger.info("Rank %d Game Seed: %d" % (rank, game_seed))
    env = create_selfplay_env(cfg, game_seed)

    agents = []
    for i in range(2):
        if cfg.common.agent[i] == 'ppo':
            agent = create_ppo_agent(cfg_process(cfg, i), env)
        elif cfg.common.agent[i] == 'dqn':
            agent = create_dqn_agent(cfg_process(cfg, i), env)
        elif cfg.common.agent[i] == 'random':
            agent = RandomAgent(action_space=env.action_space)
        elif cfg.common.agent[i] == 'keyboard':
            agent = KeyboardAgent(action_space=env.action_space)
        else:
            raise NotImplementedError
        agents.append(agent)

    value_save_path = os.path.join(cfg.common.save_path, 'selfplay_values')
    if not os.path.exists(value_save_path):
        os.mkdir(value_save_path)
    cum_return = 0.0
    action_counts = [0] * env.action_space.n
    oppo_action_counts = [0] * env.action_space.n

    obs, oppo_obs = env.reset()
    agents[0].reset()
    agents[1].reset()
    done, step_id = False, 0
    value_trace = []
    oppo_value_trace = []
    while not done:
        action = agents[0].act(obs)
        value = agents[0].value(obs)
        value_trace.append(value)
        oppo_action = agents[1].act(oppo_obs)
        oppo_value = agents[1].value(oppo_obs)
        oppo_value_trace.append(oppo_value)
        logger.info("Rank %d Step ID: %d Take Action: %d Take Oppo_action: %d" % \
                (rank, step_id, action, oppo_action))
        (obs, oppo_obs), reward, done, _ = env.step([action, oppo_action])
        action_counts[action] += 1
        oppo_action_counts[oppo_action] += 1
        cum_return += reward
        step_id += 1
    if cfg.env.save_replay:
        env.env.env.env.env.save_replay("_".join(cfg.common.agent) + "_" + \
                "_".join(cfg.model.action_type) + "_" + FLAGS.replay_path)
    path = os.path.join(value_save_path, 'value{}.pt'.format(rank))
    oppo_path = os.path.join(value_save_path, 'oppo_value{}.pt'.format(rank))
    torch.save(torch.tensor(value_trace), path)
    torch.save(torch.tensor(oppo_value_trace), oppo_path)

    for id, name in enumerate(env.action_names):
        logger.info("Rank %d\tAction ID: %d\tName: %s\tCount: %d\tOppocount: %d" %
                    (rank, id, name, action_counts[id], oppo_action_counts[id]))
    env.close()
    return cum_return

def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = []
    cfg.common.load_path.append(FLAGS.agent1_load_path)
    cfg.common.load_path.append(FLAGS.agent2_load_path)

    base_dir = os.environ.get("SC2PATH", "~/StarCraftII")
    base_dir = os.path.expanduser(base_dir)
    final_dir = base_dir + "/Replays/" + "_".join(cfg.common.agent) + \
        "_" + "_".join(cfg.model.action_type) + "_" + FLAGS.replay_path
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)

    eval_func = partial(evaluate, cfg=cfg)
    pool = Pool(min(cfg.common.num_episodes, 20))
    var_list = []
    for i in range(cfg.common.num_episodes):
        seed = random.randint(0, math.pow(2, 32) - 1)
        var_list.append({'rank': i, 'game_seed': seed})

    reward_list = pool.map(eval_func, var_list)
    print(reward_list)
    pool.close()

    print("Evaluated %d Episodes Agent1 Avg Return %f Avg Winning Rate %f" % (
        cfg.common.num_episodes, sum(reward_list) / len(reward_list),
        ((sum(reward_list) / len(reward_list)) + 1) / 2.0))

if __name__ == '__main__':
    app.run(main)
