from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import yaml
from easydict import EasyDict
import random
import torch

from absl import app
from absl import flags
from absl import logging

from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.agents.random_agent import RandomAgent
from sc2learner.agents.keyboard_agent import KeyboardAgent
from sc2learner.agents.ppo_policies_pytorch import LstmPolicy, MlpPolicy
from sc2learner.agents.rl_agent import PpoAgent
from sc2learner.utils import build_logger


FLAGS = flags.FLAGS
flags.DEFINE_string("job_name", "", "actor or learner")
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.FLAGS(sys.argv)


def create_env(cfg, random_seed=None):
    env = SC2RawEnv(map_name=cfg.env.map_name,
                    step_mul=cfg.env.step_mul,
                    agent_race='zerg',
                    bot_race='zerg',
                    difficulty=cfg.env.bot_difficulties,
                    disable_fog=cfg.env.disable_fog,
                    random_seed=random_seed)
    env = ZergActionWrapper(env,
                            game_version=cfg.env.game_version,
                            mask=cfg.env.use_action_mask,
                            use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergObservationWrapper(
        env,
        use_spatial_features=False,
        use_game_progress=(not cfg.model.policy == 'lstm'),
        action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
        use_regions=cfg.env.use_region_features)
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

    policy_func = {'mlp': MlpPolicy,
                   'lstm': LstmPolicy}
    model = policy_func[cfg.model.policy](
                ob_space=env.observation_space,
                ac_space=env.action_space,
            )
    agent = PpoAgent(env=env, model=model, cfg=cfg)
    return agent


def evaluate(cfg):
    game_seed = random.randint(0, 2**32 - 1)
    logger, _, _ = build_logger(cfg, name='evaluate')
    logger.info('cfg: {}'.format(cfg))
    logger.info("Game Seed: %d" % game_seed)
    env = create_env(cfg, game_seed)

    if cfg.common.agent == 'ppo':
        agent = create_ppo_agent(cfg, env)
    elif cfg.common.agent == 'dqn':
        agent = create_dqn_agent(cfg, env)
    elif cfg.common.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif cfg.common.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    value_save_path = os.path.join(cfg.common.save_path, 'values')
    if not os.path.exists(value_save_path):
        os.mkdir(value_save_path)
    try:
        cum_return = 0.0
        action_counts = [0] * env.action_space.n
        for i in range(cfg.common.num_episodes):
            observation = env.reset()
            agent.reset()
            done, step_id = False, 0
            value_trace = []
            while not done:
                action = agent.act(observation)
                value = agent.value(observation)
                value_trace.append(value)
                logger.info("Step ID: %d	Take Action: %d" % (step_id, action))
                observation, reward, done, _ = env.step(action)
                action_counts[action] += 1
                cum_return += reward
                step_id += 1
            if cfg.env.save_replay:
                env.env.env.save_replay(cfg.common.agent)
            path = os.path.join(value_save_path, 'value{}.pt'.format(i))
            torch.save(torch.tensor(value_trace), path)
            for id, name in enumerate(env.action_names):
                logger.info("Action ID: %d    Count: %d   Name: %s" %
                            (id, action_counts[id], name))
            logger.info("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f" % (
                i + 1, cfg.common.num_episodes, cum_return / (i + 1),
                ((cum_return / (i + 1)) + 1) / 2.0))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    evaluate(cfg)


if __name__ == '__main__':
    app.run(main)
