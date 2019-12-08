import sys
import os
import yaml
from easydict import EasyDict
import random
import time
import argparse

from sc2learner.agents.ppo_policies_pytorch import LstmPolicy, MlpPolicy
from sc2learner.agents.rl_actor import PpoActor
from sc2learner.agents.rl_learner import PpoLearner
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.rewards.reward_wrappers import KillingRewardWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper

from absl import flags
from absl import logging
from absl import app


FLAGS = flags.FLAGS
flags.DEFINE_string("job_name", "", "actor or learner")
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.FLAGS(sys.argv)


def create_env(cfg, difficulty, random_seed=None):
    env = SC2RawEnv(map_name='AbyssalReef',
                    step_mul=cfg.env.step_mul,
                    resolution=16,
                    agent_race='zerg',
                    bot_race='zerg',
                    difficulty=difficulty,
                    disable_fog=cfg.env.disable_fog,
                    tie_to_lose=False,
                    game_steps_per_episode=cfg.env.game_steps_per_episode,
                    random_seed=random_seed)
    if cfg.env.use_reward_shaping:
        env = KillingRewardWrapper(env)
    env = ZergActionWrapper(env,
                            game_version=cfg.env.game_version,
                            mask=cfg.env.use_action_mask,
                            use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergObservationWrapper(env,
                                 use_spatial_features=False,
                                 use_game_progress=(not cfg.model.policy == 'lstm'),
                                 action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
                                 use_regions=cfg.env.use_region_features)
    return env


def start_actor(cfg):
    random.seed(time.time())
    difficulty = random.choice(cfg.env.bot_difficulties.split(','))
    game_seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d Difficulty: %s" % (game_seed, difficulty))
    env = create_env(cfg, difficulty, game_seed)
    policy_func = {'mlp': MlpPolicy,
                   'lstm': LstmPolicy}
    model = policy_func[cfg.model.policy](
                ob_space=env.observation_space,
                ac_space=env.action_space,
            )
    actor = PpoActor(env, model, cfg)
    actor.run()
    env.close()


def start_learner(cfg):
    env = create_env(cfg, '1', 0)
    policy_func = {'mlp': MlpPolicy,
                   'lstm': LstmPolicy}
    model = policy_func[cfg.model.policy](
                ob_space=env.observation_space,
                ac_space=env.action_space,
            )
    learner = PpoLearner(env, model, cfg)
    learner.logger.info('cfg:{}'.format(cfg))
    learner.run()
    env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.model.load_path = FLAGS.load_path
    if FLAGS.job_name == 'actor':
        start_actor(cfg)
    elif FLAGS.job_name == 'learner':
        start_learner(cfg)
    else:
        raise ValueError


if __name__ == '__main__':
    app.run(main)
