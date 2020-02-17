import sys
import os
import yaml
from easydict import EasyDict
import random
import time
import pickle

from sc2learner.utils import ManagerZmq
from sc2learner.agents.model import PPOLSTM, PPOMLP
from sc2learner.agents.solver import PpoActor
from sc2learner.agents.solver import PpoLearner
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
flags.DEFINE_string("data_load_path", "", "path to load offline data")
flags.DEFINE_string("node_name", "", "name of the running node")
flags.DEFINE_string("seed", "", "game and network init seed for this worker")
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
    env.difficulty = difficulty
    return env


def start_actor(cfg):
    difficulty = random.choice(cfg.env.bot_difficulties.split(','))
    if 'seed' in cfg:
        seed = cfg.seed
    else:
        random.seed(time.time())
        seed = random.randint(0, 2**32 - 1)
    print("Game Seed: %d Difficulty: %s" % (seed, difficulty))
    env = create_env(cfg, difficulty, seed)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=env.observation_space,
        ac_space=env.action_space,
        seed=cfg.seed
    )
    actor = PpoActor(env, model, cfg)
    actor.run()
    env.close()


def start_learner(cfg):
    if 'seed' in cfg:
        seed = cfg.seed
    else:
        seed = 0
    ob_path = cfg.common.save_path + '/obs.pickle'
    ac_path = cfg.common.save_path + '/acs.pickle'
    try:
        with open(ob_path, 'rb') as ob:
            observation_space = pickle.load(ob)
        with open(ac_path, 'rb') as ac:
            action_space = pickle.load(ac)
        env = None
    except FileNotFoundError:
        print('Loading saved observation and action space failed, getting from env')
        env = create_env(cfg, '1', seed)
        observation_space = env.observation_space
        action_space = env.action_space
        with open(ob_path, 'wb') as ob:
            pickle.dump(observation_space, ob)
        with open(ac_path, 'wb') as ac:
            pickle.dump(action_space, ac)
    policy_func = {'mlp': PPOMLP,
                   'lstm': PPOLSTM}
    model = policy_func[cfg.model.policy](
        ob_space=observation_space,
        ac_space=action_space,
        seed=cfg.seed
    )
    learner = PpoLearner(env, model, cfg)
    learner.logger.info('cfg:{}'.format(cfg))
    learner.run()
    env.close()


def start_actor_manager(cfg):
    ip = cfg.communication.ip
    port = cfg.communication.port
    HWM = cfg.communication.HWM.actor_manager
    send_queue_size = cfg.communication.queue_size.actor_manager_send
    receive_queue_size = cfg.communication.queue_size.actor_manager_receive
    if ip.learner_manager == 'auto':
        learner_manager_ip_prefix = '.'.join(ip.learner.split('.')[0:3])
        ip.learner_manager = ip.manager_node[learner_manager_ip_prefix]
    print('auto set learner_manager ip to ' + ip.learner_manager)
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


def main(argv):
    logging.set_verbosity(logging.WARNING)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.common.data_load_path = FLAGS.data_load_path
    if FLAGS.job_name == 'actor':
        # preprocess the seed passed in
        if FLAGS.seed:
            random.seed(FLAGS.seed)
            cfg.seed = random.randint(0, 2**32 - 1)
        cfg.communication.ip.actor = '.'.join(FLAGS.node_name.split('-')[-4:])
        start_actor(cfg)
    elif FLAGS.job_name == 'learner':
        cfg.seed = cfg.train.learner_seed
        start_learner(cfg)
    elif FLAGS.job_name == 'learner_manager':
        start_learner_manager(cfg)
    elif FLAGS.job_name == 'actor_manager':
        start_actor_manager(cfg)
    else:
        raise ValueError


if __name__ == '__main__':
    app.run(main)
