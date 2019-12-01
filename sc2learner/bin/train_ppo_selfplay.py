from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread
import os
import multiprocessing
import random
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from sc2learner.agents.ppo_policies import LstmPolicy, MlpPolicy
from sc2learner.agents.ppo_agent import PPOActor, PPOLearner, PPOSelfplayActor
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.selfplay_raw_env import SC2SelfplayRawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergPlayerObservationWrapper
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job_name", 'actor', ['actor', 'learner', 'eval', 'eval_model'],
                  "Job type.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_integer("unroll_length", 128, "Length of rollout steps.")
flags.DEFINE_integer("model_cache_size", 300, "Opponent model cache size.")
flags.DEFINE_float("model_cache_prob", 0.05, "Opponent model cache probability.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_string("port_A", "5700", "Port for transporting model.")
flags.DEFINE_string("port_B", "5701", "Port for transporting data.")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_float("discount_gamma", 0.998, "Discount factor.")
flags.DEFINE_float("lambda_return", 0.95, "Lambda return factor.")
flags.DEFINE_float("clip_range", 0.1, "Clip range for PPO.")
flags.DEFINE_float("ent_coef", 0.01, "Coefficient for the entropy term.")
flags.DEFINE_float("vf_coef", 0.5, "Coefficient for the value loss.")
flags.DEFINE_float("learn_act_speed_ratio", 0, "Maximum learner/actor ratio.")
flags.DEFINE_integer("game_steps_per_episode", 43200, "Maximum steps per episode.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("learner_queue_size", 1024, "Size of learner's unroll queue.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")
flags.DEFINE_string("init_model_path", None, "Initial model path.")
flags.DEFINE_string("init_oppo_pool_filelist", None, "Initial opponent model path.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_interval", 50000, "Model saving frequency.")
flags.DEFINE_integer("print_interval", 1000, "Print train cost frequency.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use region-wise combat.")
flags.FLAGS(sys.argv)


def tf_config(ncpu=None):
  if ncpu is None:
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()


def create_env(difficulty, random_seed=None):
  env = SC2RawEnv(map_name='AbyssalReef',
                  step_mul=FLAGS.step_mul,
                  resolution=16,
                  agent_race='zerg',
                  bot_race='zerg',
                  difficulty=difficulty,
                  disable_fog=FLAGS.disable_fog,
                  tie_to_lose=False,
                  game_steps_per_episode=FLAGS.game_steps_per_episode,
                  random_seed=random_seed)
  env = ZergActionWrapper(env,
                          game_version=FLAGS.game_version,
                          mask=FLAGS.use_action_mask,
                          use_all_combat_actions=FLAGS.use_all_combat_actions)
  env = ZergObservationWrapper(env,
                               use_spatial_features=False,
                               use_game_progress=(not FLAGS.policy == 'lstm'),
                               action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
                               use_regions=FLAGS.use_region_features)
  print(env.observation_space, env.action_space)
  return env


def create_selfplay_env(random_seed=None):
  env = SC2SelfplayRawEnv(map_name='AbyssalReef',
                          step_mul=FLAGS.step_mul,
                          resolution=16,
                          agent_race='zerg',
                          opponent_race='zerg',
                          tie_to_lose=False,
                          disable_fog=FLAGS.disable_fog,
                          game_steps_per_episode=FLAGS.game_steps_per_episode,
                          random_seed=random_seed)
  env = ZergPlayerActionWrapper(
      player=0,
      env=env,
      game_version=FLAGS.game_version,
      mask=FLAGS.use_action_mask,
      use_all_combat_actions=FLAGS.use_all_combat_actions)
  env = ZergPlayerObservationWrapper(
      player=0,
      env=env,
      use_spatial_features=False,
      use_game_progress=(not FLAGS.policy == 'lstm'),
      action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
      use_regions=FLAGS.use_region_features)

  env = ZergPlayerActionWrapper(
      player=1,
      env=env,
      game_version=FLAGS.game_version,
      mask=FLAGS.use_action_mask,
      use_all_combat_actions=FLAGS.use_all_combat_actions)
  env = ZergPlayerObservationWrapper(
      player=1,
      env=env,
      use_spatial_features=False,
      use_game_progress=(not FLAGS.policy == 'lstm'),
      action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
      use_regions=FLAGS.use_region_features)
  print(env.observation_space, env.action_space)
  return env


def start_actor():
  tf_config(ncpu=2)
  random.seed(time.time())
  game_seed =  random.randint(0, 2**32 - 1)
  print("Game Seed: %d" % game_seed)
  env = create_selfplay_env(game_seed)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  actor = PPOSelfplayActor(
      env=env,
      policy=policy,
      unroll_length=FLAGS.unroll_length,
      gamma=FLAGS.discount_gamma,
      lam=FLAGS.lambda_return,
      model_cache_size=FLAGS.model_cache_size,
      model_cache_prob=FLAGS.model_cache_prob,
      prob_latest_opponent=0.0,
      init_opponent_pool_filelist=FLAGS.init_oppo_pool_filelist,
      freeze_opponent_pool=False,
      learner_ip=FLAGS.learner_ip,
      port_A=FLAGS.port_A,
      port_B=FLAGS.port_B)
  actor.run()
  env.close()


def start_learner():
  tf_config()
  env = create_env('1', 0)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  learner = PPOLearner(env=env,
                       policy=policy,
                       unroll_length=FLAGS.unroll_length,
                       lr=FLAGS.learning_rate,
                       clip_range=FLAGS.clip_range,
                       batch_size=FLAGS.batch_size,
                       ent_coef=FLAGS.ent_coef,
                       vf_coef=FLAGS.vf_coef,
                       max_grad_norm=0.5,
                       queue_size=FLAGS.learner_queue_size,
                       print_interval=FLAGS.print_interval,
                       save_interval=FLAGS.save_interval,
                       learn_act_speed_ratio=FLAGS.learn_act_speed_ratio,
                       save_dir=FLAGS.save_dir,
                       init_model_path=FLAGS.init_model_path,
                       port_A=FLAGS.port_A,
                       port_B=FLAGS.port_B)
  learner.run()
  env.close()


def start_evaluator_against_builtin():
  tf_config(ncpu=2)
  random.seed(time.time())
  difficulty = random.choice(FLAGS.difficulties.split(','))
  game_seed =  random.randint(0, 2**32 - 1)
  print("Game Seed: %d" % game_seed)
  env = create_env(difficulty, game_seed)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  actor = PPOActor(env=env,
                   policy=policy,
                   unroll_length=FLAGS.unroll_length,
                   gamma=FLAGS.discount_gamma,
                   lam=FLAGS.lambda_return,
                   enable_push=False,
                   learner_ip=FLAGS.learner_ip,
                   port_A=FLAGS.port_A,
                   port_B=FLAGS.port_B)
  actor.run()
  env.close()


def start_evaluator_against_model():
  tf_config(ncpu=2)
  random.seed(time.time())
  game_seed =  random.randint(0, 2**32 - 1)
  print("Game Seed: %d" % game_seed)
  env = create_selfplay_env(game_seed)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  actor = PPOSelfplayActor(
      env=env,
      policy=policy,
      unroll_length=FLAGS.unroll_length,
      gamma=FLAGS.discount_gamma,
      lam=FLAGS.lambda_return,
      model_cache_size=1,
      model_cache_prob=FLAGS.model_cache_prob,
      enable_push=False,
      prob_latest_opponent=0.0,
      init_opponent_pool_filelist=FLAGS.init_oppo_pool_filelist,
      freeze_opponent_pool=True,
      learner_ip=FLAGS.learner_ip,
      port_A=FLAGS.port_A,
      port_B=FLAGS.port_B)
  actor.run()
  env.close()


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  if FLAGS.job_name == 'actor': start_actor()
  elif FLAGS.job_name == 'learner': start_learner()
  elif FLAGS.job_name == 'eval': start_evaluator_against_builtin()
  else: start_evaluator_against_model()


if __name__ == '__main__':
  app.run(main)
