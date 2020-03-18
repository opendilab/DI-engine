from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import traceback
import multiprocessing

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from sc2learner.envs.lan_raw_env import LanSC2RawEnv
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.utils.utils import print_arguments
from sc2learner.agents.ppo_policies import LstmPolicy, MlpPolicy
from sc2learner.agents.ppo_agent import PPOAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_string("model_path", None, "Filepath to load initial model.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("host", "127.0.0.1", "Game Host. Can be 127.0.0.1 or ::1")
flags.DEFINE_integer(
    "config_port", 14380, "Where to set/find the config port. The host starts a tcp server to share "
    "the config with the client, and to proxy udp traffic if played over an "
    "ssh tunnel. This sets that port, and is also the start of the range of "
    "ports used for LAN play."
)
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use action mask or not.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")


def print_actions(env):
    print("----------------------------- Actions -----------------------------")
    for action_id, action_name in enumerate(env.action_names):
        print("Action ID: %d	Action Name: %s" % (action_id, action_name))
    print("-------------------------------------------------------------------")


def print_action_distribution(env, action_counts):
    print("----------------------- Action Distribution -----------------------")
    for action_id, action_name in enumerate(env.action_names):
        print("Action ID: %d	Count: %d	Name: %s" % (action_id, action_counts[action_id], action_name))
    print("-------------------------------------------------------------------")


def tf_config(ncpu=None):
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin':
            ncpu //= 2
    config = tf.ConfigProto(
        allow_soft_placement=True, intra_op_parallelism_threads=ncpu, inter_op_parallelism_threads=ncpu
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


def start_lan_agent():
    """Run the agent, connecting to a host started independently."""
    tf_config()
    env = LanSC2RawEnv(
        host=FLAGS.host,
        config_port=FLAGS.config_port,
        agent_race='zerg',
        step_mul=FLAGS.step_mul,
        visualize_feature_map=False
    )
    env = ZergActionWrapper(
        env,
        game_version=FLAGS.game_version,
        mask=FLAGS.use_action_mask,
        use_all_combat_actions=FLAGS.use_all_combat_actions
    )
    env = ZergObservationWrapper(
        env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features
    )
    print_actions(env)
    policy = {'lstm': LstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    agent = PPOAgent(env=env, policy=policy, model_path=FLAGS.model_path)
    try:
        action_counts = [0] * env.action_space.n
        observation = env.reset()
        done, step_id = False, 0
        while not done:
            action = agent.act(observation)
            print("Step ID: %d	Take Action: %d" % (step_id, action))
            observation, reward, done, _ = env.step(action)
            action_counts[action] += 1
            step_id += 1
        print_action_distribution(env, action_counts)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    env.close()


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)
    print_arguments(FLAGS)
    start_lan_agent()


if __name__ == "__main__":
    app.run(main)
