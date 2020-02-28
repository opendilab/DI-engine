import sys
import os
import yaml
from easydict import EasyDict
import random
import time
import pickle

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


def start_actor(cfg):
    from sc2learner.agents.solver import PpoActor
    actor = PpoActor(cfg)
    actor.run()


def start_learner(cfg):
    from sc2learner.agents.model import PPOLSTM, PPOMLP
    from sc2learner.agents.solver import PpoLearner, create_env
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
        env = create_env(cfg, '1', cfg.train.learner_seed)
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
        seed=cfg.train.learner_seed
    )
    learner = PpoLearner(env, model, cfg)
    learner.logger.info('cfg:{}'.format(cfg))
    learner.run()
    env.close()


def start_actor_manager(cfg):
    '''
    Fully functional actor manager for relaying both model(from learner to actor)
    and trajectory(from actor to learner)
    '''

    from sc2learner.utils import ManagerZmq
    ip = cfg.communication.ip
    port = cfg.communication.port
    HWM = cfg.communication.HWM.actor_manager
    send_queue_size = cfg.communication.queue_size.actor_manager_send
    receive_queue_size = cfg.communication.queue_size.actor_manager_receive
    if ip.learner_manager == 'auto':
        learner_manager_ip_prefix = '.'.join(ip.learner.split('.')[0:3])
        ip.learner_manager = ip.manager_node[learner_manager_ip_prefix]
    print('auto set learner_manager ip to ' + ip.learner_manager)
    if ip.coordinator == 'learner_manager':
        ip.coordinator = ip.learner_manager
    print('IP address of coordinator is set to the learner_manager')
    apply_ip = {
        'send': ip.learner_manager,
        'relay': ip.coordinator
    }
    apply_port = {
        'send': port.learner_manager,
        'receive': port.actor_manager,
        'request': port.actor_manager_model,
        'reply': port.actor_model,
        'relay_in': port.coordinator_relayed,
        'relay_out': port.coordinator
    }
    time_interval = cfg.communication.model_time_interval
    manager = ManagerZmq(apply_ip, apply_port, name='actor_manager', HWM=HWM,
                         send_queue_size=send_queue_size, receive_queue_size=receive_queue_size,
                         time_interval=time_interval)
    manager.run({'sender': True, 'receiver': True,
                 'forward_request': True, 'forward_reply': True, 'relay': True})


def start_coordinator(cfg):
    from sc2learner.utils import Coordinator
    if 'IN_K8S' in os.environ:
        port = os.getenv('SENSESTAR_COORDINATOR_SERVICE_SERVICE_PORT_JOB')
    else:
        port = cfg.communication.port.coordinator
    coordinator = Coordinator(cfg, port)
    coordinator.run()


def start_actor_model_manager(cfg):
    from sc2learner.utils import ManagerZmq
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
    }
    apply_port = {
        'request': port.actor_manager_model,
        'reply': port.actor_model,
    }
    time_interval = cfg.communication.model_time_interval
    manager = ManagerZmq(apply_ip, apply_port, name='actor_model_manager', HWM=HWM,
                         send_queue_size=send_queue_size, receive_queue_size=receive_queue_size,
                         time_interval=time_interval)
    manager.run({'sender': False, 'receiver': False,
                 'forward_request': True, 'forward_reply': True, 'relay': False})


def start_actor_data_manager(cfg):
    from sc2learner.utils import ManagerZmq
    ip = cfg.communication.ip
    port = cfg.communication.port
    HWM = cfg.communication.HWM.actor_manager
    send_queue_size = cfg.communication.queue_size.actor_manager_send
    receive_queue_size = cfg.communication.queue_size.actor_manager_receive
    if ip.learner_manager == 'auto':
        learner_manager_ip_prefix = '.'.join(ip.learner.split('.')[0:3])
        ip.learner_manager = ip.manager_node[learner_manager_ip_prefix]
    print('auto set learner_manager ip to ' + ip.learner_manager)
    if ip.coordinator == 'learner_manager':
        ip.coordinator = ip.learner_manager
    print('IP address of coordinator is set to the learner_manager')
    apply_ip = {
        'send': ip.learner_manager,
        'relay': ip.coordinator
    }
    apply_port = {
        'send': port.learner_manager,
        'receive': port.actor_manager,
        'relay_in': port.coordinator_relayed,
        'relay_out': port.coordinator
    }
    time_interval = cfg.communication.model_time_interval
    manager = ManagerZmq(apply_ip, apply_port, name='actor_data_manager', HWM=HWM,
                         send_queue_size=send_queue_size, receive_queue_size=receive_queue_size,
                         time_interval=time_interval)
    manager.run({'sender': True, 'receiver': True,
                 'forward_request': False, 'forward_reply': False, 'relay': True})


def start_learner_manager(cfg):
    from sc2learner.utils import ManagerZmq

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
    manager.run({'sender': True, 'receiver': True, 'relay': False,
                 'forward_request': True, 'forward_reply': True})


def main(argv):
    logging.set_verbosity(logging.WARNING)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    if 'IN_K8S' in os.environ:
        cfg.common.save_path = os.getenv('SAVE_PATH', '')
        cfg.common.load_path = os.getenv('LOAD_PATH', '')
        cfg.common.data_load_path = os.getenv('DATA_LOAD_PATH', '')
    else:
        cfg.common.save_path = os.path.dirname(FLAGS.config_path)
        cfg.common.load_path = FLAGS.load_path
        cfg.common.data_load_path = FLAGS.data_load_path
    if FLAGS.job_name == 'actor':
        # preprocess the seed passed in
        if FLAGS.seed:
            random.seed(FLAGS.seed)
            cfg.seed = random.randint(0, 2**32 - 1)
        if 'IN_K8S' not in os.environ:
            cfg.communication.ip.actor = '.'.join(FLAGS.node_name.split('-')[-4:])
        start_actor(cfg)
    elif FLAGS.job_name == 'learner':
        start_learner(cfg)
    elif FLAGS.job_name == 'learner_manager':
        start_learner_manager(cfg)
    elif FLAGS.job_name == 'actor_manager':
        start_actor_manager(cfg)
    elif FLAGS.job_name == 'actor_data_manager':
        start_actor_data_manager(cfg)
    elif FLAGS.job_name == 'actor_model_manager':
        start_actor_model_manager(cfg)
    elif FLAGS.job_name == 'coordinator':
        start_coordinator(cfg)
    else:
        raise ValueError


if __name__ == '__main__':
    app.run(main)
