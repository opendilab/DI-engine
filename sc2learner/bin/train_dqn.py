from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import random
import time

import torch
from absl import app
from absl import flags
from absl import logging

from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.agents.dqn_agent import DQNActor
from sc2learner.agents.dqn_agent import DQNLearner
from sc2learner.agents.dqn_networks import NonspatialDuelingQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job_name", 'actor', ['actor', 'learner'], "Job type.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_string("ports", "5700,5701,5702",
                    "3 ports for distributed replay memory.")
flags.DEFINE_integer("client_memory_size", 50000,
                     "Total size of client memory.")
flags.DEFINE_integer("client_memory_warmup_size", 2000,
                     "Memory warmup size for client.")
flags.DEFINE_integer("server_memory_size", 1000000,
                     "Total size of server memory.")
flags.DEFINE_integer("server_memory_warmup_size", 100000,
                     "Memory warmup size for client.")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_float("discount", 0.995, "Discount factor.")
flags.DEFINE_float("send_freq", 4.0, "Probability of a step being pushed.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay_steps", 1000000, "Greedy epsilon decay step.")
flags.DEFINE_integer("eps_decay_steps2", 10000000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("mmc_beta", 0.9, "Discount.")
flags.DEFINE_integer("target_update_interval", 10000,
                     "Target net update interval.")
flags.DEFINE_string("init_model_path", None, "Checkpoint to initialize model.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Dir to save models to")
flags.DEFINE_integer("checkpoint_interval", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_interval", 10000, "Print train cost frequency.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", True, "Use region features")
flags.FLAGS(sys.argv)


def create_env(difficulty, random_seed=None):
  env = SC2RawEnv(map_name='AbyssalReef',
                   step_mul=FLAGS.step_mul,
                   resolution=16,
                   agent_race='zerg',
                   bot_race='zerg',
                   difficulty=difficulty,
                   disable_fog=FLAGS.disable_fog,
                   random_seed=random_seed)
  env = ZergActionWrapper(env,
                          game_version=FLAGS.game_version,
                          mask=False,
                          use_all_combat_actions=FLAGS.use_all_combat_actions)
  env = ZergObservationWrapper(env,
                               use_spatial_features=False,
                               use_regions=FLAGS.use_region_features)
  return env


def create_network(env):
  return NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                               n_out=env.action_space.n)


def start_actor_job():
  random.seed(time.time())
  difficulty = random.choice(FLAGS.difficulties.split(','))
  game_seed =  random.randint(0, 2**32 - 1)
  print("Game Seed: %d Difficulty: %s" % (game_seed, difficulty))
  env = create_env(difficulty, game_seed)
  network = create_network(env)
  actor = DQNActor(memory_size=FLAGS.client_memory_size,
                   memory_warmup_size=FLAGS.client_memory_warmup_size,
                   env=env,
                   network=network,
                   discount=FLAGS.discount,
                   send_freq=FLAGS.send_freq,
                   ports=FLAGS.ports.split(','),
                   learner_ip=FLAGS.learner_ip)
  actor.run()
  env.close()


def start_learner_job():
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  env = create_env('1', 0)
  network = create_network(env)
  learner = DQNLearner(network=network,
                       action_space=env.action_space,
                       memory_size=FLAGS.server_memory_size,
                       memory_warmup_size=FLAGS.server_memory_warmup_size,
                       discount=FLAGS.discount,
                       eps_start=FLAGS.eps_start,
                       eps_end=FLAGS.eps_end,
                       eps_decay_steps=FLAGS.eps_decay_steps,
                       eps_decay_steps2=FLAGS.eps_decay_steps2,
                       batch_size=FLAGS.batch_size,
                       mmc_beta=FLAGS.mmc_beta,
                       gradient_clipping=FLAGS.gradient_clipping,
                       adam_eps=FLAGS.adam_eps,
                       learning_rate=FLAGS.learning_rate,
                       target_update_interval=FLAGS.target_update_interval,
                       checkpoint_dir=FLAGS.checkpoint_dir,
                       checkpoint_interval=FLAGS.checkpoint_interval,
                       print_interval=FLAGS.print_interval,
                       ports=FLAGS.ports.split(','),
                       init_model_path=FLAGS.init_model_path)
  learner.run()
  env.close()


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  if FLAGS.job_name == 'actor': start_actor_job()
  else: start_learner_job()


if __name__ == '__main__':
  app.run(main)
