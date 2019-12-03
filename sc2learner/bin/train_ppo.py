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

from sc2learner.agents.ppo_policies_pytorch import LstmPolicy, MlpPolicy
from sc2learner.agents.rl_actor import PpoActor
from sc2learner.agents.rl_learner import PpoLearner
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.rewards.reward_wrappers import KillingRewardWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job_name", 'actor', ['actor', 'learner'], "Job type.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_integer("unroll_length", 128, "Length of rollout steps.")
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
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("game_steps_per_episode", 43200,
                     "Maximum steps per episode.")
flags.DEFINE_integer("learner_queue_size", 1024,
                     "Size of learner's unroll queue.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")
flags.DEFINE_string("init_model_path", None, "Initial model path.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_interval", 50000, "Model saving frequency.")
flags.DEFINE_integer("print_interval", 1000, "Print train cost frequency.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use region-wise combat.")
flags.DEFINE_boolean("use_reward_shaping", False, "Use reward shaping.")
flags.FLAGS(sys.argv)


'''
def tf_config(ncpu=None):
  if ncpu is None:
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()
'''


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
    if FLAGS.use_reward_shaping:
        env = KillingRewardWrapper(env)
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


def start_actor():
    random.seed(time.time())
    difficulty = random.choice(FLAGS.difficulties.split(','))
    game_seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d Difficulty: %s" % (game_seed, difficulty))
    env = create_env(difficulty, game_seed)
    port = {}
    port['actor'] = "5700"
    port['learner'] = "5701"
    model = MlpPolicy(
                ob_space=env.observation_space,
                ac_space=env.action_space,
            )
    actor = PpoActor(env=env,
                     model=model,
                     unroll_length=FLAGS.unroll_length,
                     enable_push=True,
                     queue_size=1,
                     gamma=FLAGS.discount_gamma,
                     lam=FLAGS.lambda_return,
                     learner_ip=FLAGS.learner_ip,
                     port=port)
    actor.run()
    env.close()


def start_learner():
    env = create_env('1', 0)
    port = {}
    port['actor'] = "5700"
    port['learner'] = "5701"
    model = MlpPolicy(
                ob_space=env.observation_space,
                ac_space=env.action_space,
            )
    learner = PpoLearner(env=env,
                         model=model,
                         unroll_length=FLAGS.unroll_length,
                         #lr=FLAGS.learning_rate,
                         batch_size=FLAGS.batch_size,
                         entropy_coeff=FLAGS.ent_coef,
                         value_coeff=FLAGS.vf_coef,
                         # max_grad_norm=0.5,
                         queue_size=FLAGS.learner_queue_size,
                         unroll_split=128,
                         # print_interval=FLAGS.print_interval,
                         # save_interval=FLAGS.save_interval,
                         #learn_act_speed_ratio=FLAGS.learn_act_speed_ratio,
                         # save_dir=FLAGS.save_dir,
                         # init_model_path=FLAGS.init_model_path,
                         port=port)
    learner.run()
    env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    print_arguments(FLAGS)
    if FLAGS.job_name == 'actor':
        start_actor()
    else:
        start_learner()


if __name__ == '__main__':
    app.run(main)
