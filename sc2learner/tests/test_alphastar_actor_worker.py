"""
Unit test for the AlphaStar actor worker (not including model&stat loading)
"""
import random
import time
import os
import pytest
import socket
from threading import Thread
import logging
import tempfile
import warnings
import atexit

import yaml
import torch
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.worker.actor.alphastar_actor_worker import AlphaStarActorWorker
from sc2learner.data.fake_dataset import FakeActorDataset, get_single_step_data
from sc2learner.utils.file_helper import read_file_ceph
# TODO: move the api modules
from sc2learner.api.coordinator import Coordinator
from sc2learner.api.coordinator_api import create_coordinator_app
from sc2learner.api.manager import Manager
from sc2learner.api.manager_api import create_manager_app
from sc2learner.utils.compression_helper import get_step_data_decompressor

PRINT_ACTIONS = False
TEMP_TRAJ_DIR = tempfile.TemporaryDirectory()
atexit.register(TEMP_TRAJ_DIR.cleanup)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('config_path', '', 'Path to the config yaml file for test')
    flags.DEFINE_bool('fake_dataset', True, 'Whether to use fake dataset')
else:
    FLAGS = None
    PYTEST_FAKE_DATASET = True


class ActorForTest(AlphaStarActorWorker):
    def __init__(self, cfg):
        super(ActorForTest, self).__init__(cfg)

    def _make_env(self, players):
        if (FLAGS and FLAGS.fake_dataset) or PYTEST_FAKE_DATASET:
            from .fake_env import FakeEnv
            return FakeEnv(len(players))
        else:
            return super()._make_env(players)

    def _module_init(self):
        super()._module_init()
        self.last_time = None

    def action_modifier(self, act, step):
        # if act[0]:
        #    recu_check_keys(get_single_step_data()['actions'], act[0])
        # if act[1]:
        #    recu_check_keys(get_single_step_data()['actions'], act[1])
        if self.cfg.env.use_cuda:
            print('Max CUDA memory:{}'.format(torch.cuda.max_memory_allocated()))
        t = time.time()
        if self.last_time is not None:
            print('Time between action:{}'.format(t - self.last_time))
        self.last_time = t
        for n in range(len(act)):
            if act[n] and act[n]['delay'] == 0:
                act[n]['delay'] = random.randint(0, 10)
            if PRINT_ACTIONS:
                print('Act {}:{}'.format(n, str(act[n])))
        return act


def get_test_cfg():
    with open(__file__.replace('.py', '.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    local_ip = '127.0.0.1'  # socket.gethostname()
    learner_ip = local_ip
    coordinator_ip = local_ip
    manager_ip = local_ip
    cfg.api.learner_ip = learner_ip
    cfg.api.coordinator_ip = coordinator_ip
    cfg.api.manager_ip = manager_ip
    cfg.actor.coordinator_ip = coordinator_ip
    cfg.actor.manager_ip = manager_ip
    cfg.actor.ceph_traj_path = TEMP_TRAJ_DIR.name + os.sep
    cfg.actor.ceph_model_path = 'do_not_save/'
    cfg.actor.ceph_stat_path = 'do_not_save/'
    cfg.data = {}
    cfg.data.train = {}
    cfg.data.train.batch_size = 128
    return cfg


@pytest.fixture(scope='module')
def coordinator():
    cfg = get_test_cfg()
    coordinator = Coordinator(cfg)
    app = create_coordinator_app(coordinator)

    def run():
        app.run(host=cfg.api.coordinator_ip, port=cfg.api.coordinator_port, debug=True, use_reloader=False)

    coordinator_thread = Thread(target=run)
    coordinator_thread.daemon = True
    coordinator_thread.start()
    logging.info('coordinator started')
    yield coordinator
    coordinator.close()


@pytest.fixture(scope='module')
def manager():
    cfg = get_test_cfg()
    manager = Manager(cfg)
    app = create_manager_app(manager)

    def run():
        app.run(host=cfg.api.manager_ip, port=cfg.api.manager_port, debug=True, use_reloader=False)

    manager_thread = Thread(target=run)
    manager_thread.daemon = True
    manager_thread.start()
    logging.info('manager started')
    return manager


IGNORE_LIST = ['target_outputs', 'actions:delay']


def recu_check_keys(ref, under_test, trace='ROOT'):
    for item in IGNORE_LIST:
        if item in trace:
            print('Skipped {}'.format(trace))
            return
    # print('Checking {}'.format(trace))
    if under_test is None and ref is not None\
       or ref is None and under_test is not None:
        warnings.warn('Only one is None. REF={} DUT={} {}'.format(ref, under_test, trace))
    elif isinstance(under_test, torch.Tensor) or isinstance(ref, torch.Tensor):
        assert(isinstance(under_test, torch.Tensor) and isinstance(ref, torch.Tensor)),\
            'one is tensor and the other is not tensor or None {}'.format(trace)
        if under_test.size() != ref.size():
            warnings.warn('Mismatch size: REF={} DUT={} {}'.format(ref.size(), under_test.size(), trace))
    elif isinstance(under_test, list) or isinstance(under_test, tuple):
        if len(under_test) != len(ref):
            warnings.warn('Mismatch length: REF={} DUT={} {}'.format(len(ref), len(under_test), trace))
        for n in range(min(len(ref), len(under_test))):
            if n in IGNORE_LIST:
                print('Skipped {}'.format(trace + ':' + str(n)))
                return
            recu_check_keys(ref[n], under_test[n], trace=trace + ':' + str(n))
    elif isinstance(under_test, dict):
        assert isinstance(ref, dict)
        for k, v in ref.items():
            if k in IGNORE_LIST:
                print('Skipped {}'.format(trace + ':' + str(k)))
                return
            if k in under_test:
                recu_check_keys(v, under_test[k], trace=trace + ':' + str(k))
            else:
                warnings.warn('Missing key: {}'.format(trace + ':' + str(k)))


def check_with_fake_dataset(traj):
    fake_dataset = FakeActorDataset(trajectory_len=1)
    recu_check_keys(fake_dataset.get_1v1_agent_data(), traj[-1:])


def test_actor(coordinator, manager, caplog):
    caplog.set_level(logging.INFO)
    # to be called by pytest
    cfg = get_test_cfg()
    actor = ActorForTest(cfg)
    logging.info('expecting a manager registered at the coordinator {}'.format(str(coordinator.manager_record)))
    assert (len(coordinator.manager_record) == 1)
    logging.info('expecting a actor registered at the manager {}'.format(str(manager.actor_record)))
    assert (len(manager.actor_record) == 1)

    # Running a episode
    logging.info('actor started, running the 1st loop')
    actor.run_episode()

    logging.info('expecting a job record in coordinator: {}'.format(str(coordinator.job_record)))
    job_id = list(coordinator.job_record.keys())[0]
    assert (job_id in manager.job_record)
    # the numbers are set in the coordinator (in the fake jobs)
    logging.info(
        'expecting a job record in manager with ceil(30/8)*2=8 entries: {}'.format(
            str(len(manager.job_record[job_id]['metadatas']))
        )
    )
    # print(manager.job_record[job_id]['metadatas'])
    for entry in manager.job_record[job_id]['metadatas']:
        assert (entry['length'] == 8)
    assert (manager.job_record[job_id]['state'] == 'finish')
    assert (len(manager.job_record[job_id]['metadatas']) == 8)
    batch_size = 2
    time.sleep(0.1)
    # notice we are using a very small number of samples for testing
    # so the replay buffer caching should be set to a small value
    smpls = coordinator.replay_buffer.sample(batch_size)
    assert smpls is not None
    assert isinstance(smpls, list) and len(smpls) == batch_size
    decompressor = get_step_data_decompressor(cfg.env.compress_obs)
    for smpl in smpls:
        traj = read_file_ceph(smpl['ceph_name'] + smpl['trajectory_path'], read_type='pickle')
        decompressed_traj = decompressor(traj)
        # here I'm checking the stored trajectory against FakeDataset
        # the format of observations is not tested here as it's from the same FakeEnv and the same FakeDataset
        # we need alphastar_actor_env_test.py to test the transfer of observation from SC2 to env to actor
        # but the outputs from the model are real and tested
        check_with_fake_dataset(decompressed_traj)

    # Running another episode
    logging.info('actor running the 2nd loop')
    actor.run_episode()
    actor.heartbeat_worker.stop_heatbeat()


def main(unused_argv):
    # start a actor for full test on cluster (with network connection)
    # not used for pytest
    with open(FLAGS.config_path) as f:
        ft_cfg = yaml.load(f)
    ft_cfg = EasyDict(ft_cfg)
    ft_cfg["log_path"] = os.path.dirname(FLAGS.config_path)
    ta = ActorForTest(ft_cfg)
    ta.run()


if __name__ == '__main__':
    app.run(main)
