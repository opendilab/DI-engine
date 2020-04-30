import pytest
import copy
import os
import sys
import yaml
import socket
import signal
from easydict import EasyDict
from threading import Thread
from flask import Flask, request
from sc2learner.system import Coordinator, create_coordinator_app, LeagueManagerWrapper, create_league_manager_app
from sc2learner.worker.learner.api import LearnerCommunicationHelper
from sc2learner.utils.log_helper import TextLogger

with open(os.path.join(os.path.dirname(__file__), 'alphastar_rl_learner_default_config.yaml')) as f:
    cfg = yaml.safe_load(f)
cfg = EasyDict(cfg)
learner_ip = '127.0.0.1'  # socket.gethostname()
coordinator_ip = '127.0.0.1'  # socket.gethostname()
league_manager_ip = '127.0.0.1'
cfg.system.learner_ip = learner_ip
cfg.system.coordinator_ip = coordinator_ip
cfg.system.league_manager_ip = league_manager_ip
cfg.system.learner_port += 30
cfg.system.coordinator_port += 30

log_path = os.path.join(os.path.dirname(__file__))
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
if not os.path.exists(log_path):
    os.mkdir(log_path)


def build_ret(code, info=''):
    return {'code': code, 'info': info}


@pytest.fixture(scope='module')
def setup_config():
    return cfg


@pytest.fixture(scope='module')
def fake_train_learner():
    learner_app = Flask("__main__")

    def run():
        try:
            learner_app.run(host=learner_ip, port=cfg.system.learner_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    logger = TextLogger(log_path, name="learner.log")
    learner_cfg = copy.deepcopy(cfg)
    cfg.data = {}
    cfg.data.train = {}
    cfg.data.train.batch_size = 128
    learner = LearnerCommunicationHelper(cfg)
    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield learner
    # close resource operation


@pytest.fixture(scope='module')
def coordinator():
    logger = TextLogger(log_path, name="coordinator.log")
    coordinator = Coordinator(cfg)
    coordinator_app = create_coordinator_app(coordinator)

    def run():
        try:
            coordinator_app.run(host=coordinator_ip, port=cfg.system.coordinator_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield coordinator
    # close resource operation
    coordinator.close()


@pytest.fixture(scope='module')
def league_manager():
    logger = TextLogger(log_path, name="league_manager.log")
    league_manager_wrapper = LeagueManagerWrapper(cfg)
    app = create_league_manager_app(league_manager_wrapper)

    def run():
        try:
            app.run(host=league_manager_ip, port=cfg.system.league_manager_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield league_manager_wrapper
    # close resource operation
    league_manager_wrapper.close()
