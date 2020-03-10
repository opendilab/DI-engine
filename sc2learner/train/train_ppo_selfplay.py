from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread
import os
import yaml
from easydict import EasyDict
import multiprocessing
import random
import time

from absl import app
from absl import flags
from absl import logging

from sc2learner.utils import ManagerZmq
from sc2learner.agents.model import PPOLSTM, PPOMLP
from sc2learner.agents.solver import PpoActor, PpoLearner, PpoSelfplayActor
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.selfplay_raw_env import SC2SelfplayRawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergPlayerObservationWrapper


FLAGS = flags.FLAGS
flags.DEFINE_string("job_name", "", "actor or learner")
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("data_load_path", "", "path to load offline data")
flags.FLAGS(sys.argv)


def create_env(cfg, random_seed=None):
    env = SC2RawEnv(map_name=cfg.env.map_name,
                    step_mul=cfg.env.step_mul,
                    resolution=cfg.env.resolution,
                    agent_race=cfg.env.agent_race,
                    bot_race=cfg.env.bot_race,
                    difficulty=cfg.env.difficulty,
                    disable_fog=cfg.env.disable_fog,
                    tie_to_lose=cfg.env.tie_to_lose,
                    game_steps_per_episode=cfg.env.game_steps_per_episode,
                    random_seed=random_seed)
    env = ZergActionWrapper(env,
                            game_version=cfg.env.game_version,
                            mask=cfg.env.use_action_mask,
                            use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergObservationWrapper(env,
                                 use_spatial_features=cfg.env.use_spatial_features,
                                 use_game_progress=(not cfg.model.policy == 'lstm'),
                                 action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
                                 use_regions=cfg.env.use_region_features)
    print(env.observation_space, env.action_space)
    return env


def create_selfplay_env(cfg, random_seed=None):
    env = SC2SelfplayRawEnv(map_name=cfg.env.map_name,
                            step_mul=cfg.env.step_mul,
                            resolution=cfg.env.resolution,
                            agent_race=cfg.env.agent_race,
                            opponent_race=cfg.env.opponent_race,
                            tie_to_lose=cfg.env.tie_to_lose,
                            disable_fog=cfg.disable_fog,
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


def start_actor(cfg):
    random.seed(time.time())
    game_seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d" % game_seed)
    env = create_selfplay_env(cfg, game_seed)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=env.observation_space,
        ac_space=env.action_space,
    )
    actor = PpoSelfplayActor(env, model, cfg)
    # model_cache_size=cfg.train.model_cache_size,
    # model_cache_prob=cfg.train.model_cache_prob,
    # prob_latest_opponent=0.0,
    # init_opponent_pool_filelist=FLAGS.init_oppo_pool_filelist,
    # freeze_opponent_pool=False,
    # learner_ip=FLAGS.learner_ip,
    # port_A=FLAGS.port_A,
    # port_B=FLAGS.port_B)
    actor.run()
    env.close()


def start_learner(cfg):
    env = create_env(cfg, '1', 0)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=env.observation_space,
        ac_space=env.action_space,
    )
    learner = PpoLearner(env, model, cfg)
    learner.logger.info('cfg:{}'.format(cfg))
    # max_grad_norm=0.5,
    # learn_act_speed_ratio=FLAGS.learn_act_speed_ratio,
    # save_dir=FLAGS.save_dir,
    # init_model_path=FLAGS.init_model_path,
    # port_A=FLAGS.port_A,
    # port_B=FLAGS.port_B)
    learner.run()
    env.close()


def start_actor_manager(cfg):
    ip = cfg.communication.ip
    port = cfg.communication.port
    HWM = cfg.communication.HWM.actor_manager
    send_queue_size = cfg.communication.queue_size.actor_manager_send
    receive_queue_size = cfg.communication.queue_size.actor_manager_receive
    apply_ip = {
        'send': ip.learner_manager,
    }
    apply_port = {
        'send': port.learner_manager,
        'receive': port.actor_manager,
        'request': port.actor_manager_model,
        'reply': port.actor_model,
    }
    time_interval = cfg.communication.model_time_interval
    manager = ManagerZmq(apply_ip, apply_port, name='actor_manager', HWM=HWM,
                         send_queue_size=send_queue_size, receive_queue_size=receive_queue_size,
                         time_interval=time_interval)
    manager.run({'sender': True, 'receiver': True,
                 'forward_request': True, 'forward_reply': True})


def start_learner_manager(cfg):
    ip = cfg.communication.ip
    port = cfg.communication.port
    HWM = cfg.communication.HWM.learner_manager
    send_queue_size = cfg.communication.queue_size.learner_manager_send
    receive_queue_size = cfg.communication.queue_size.learner_manager_receive
    apply_ip = {
        'send': ip.learner,
    }
    apply_port = {
        'send': port.learner,
        'receive': port.learner_manager,
        'request': port.learner_manager_model,
        'reply': port.actor_manager_model,
    }
    time_interval = cfg.communication.model_time_interval
    manager = ManagerZmq(apply_ip, apply_port, name='learner_manager', HWM=HWM,
                         send_queue_size=send_queue_size, receive_queue_size=receive_queue_size,
                         time_interval=time_interval)
    manager.run({'sender': True, 'receiver': True,
                 'forward_request': True, 'forward_reply': True})


def start_evaluator_against_builtin(cfg):
    random.seed(time.time())
    game_seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d" % game_seed)
    env = create_env(cfg, game_seed)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=env.observation_space,
        ac_space=env.action_space,
    )
    actor = PpoActor(env, model, cfg)
    actor.run()
    env.close()


def start_evaluator_against_model(cfg):
    random.seed(time.time())
    game_seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d" % game_seed)
    env = create_selfplay_env(game_seed)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=env.observation_space,
        ac_space=env.action_space,
    )
    actor = PpoSelfplayActor(env, model, cfg)
    actor.run()
    env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.common.data_load_path = FLAGS.data_load_path

    if FLAGS.job_name == 'actor':
        start_actor(cfg)
    elif FLAGS.job_name == 'learner':
        start_learner(cfg)
    elif FLAGS.job_name == 'learner_manager':
        start_learner_manager(cfg)
    elif FLAGS.job_name == 'actor_manager':
        start_actor_manager(cfg)
    elif FLAGS.job_name == 'eval_buildin':
        start_evaluator_against_builtin(cfg)
    elif FLAGS.job_name == 'eval_model':
        start_evaluator_against_model(cfg)
    else:
        raise ValueError


if __name__ == '__main__':
    app.run(main)
