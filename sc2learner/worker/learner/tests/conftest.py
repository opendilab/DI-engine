import pytest
import os
import sys
import yaml
import socket
import signal
from easydict import EasyDict
from threading import Thread
from flask import Flask, request
from sc2learner.api import Coordinator, LearnerCommunicationHelper
from sc2learner.api.coordinator_api import create_coordinator_app
from sc2learner.utils.log_helper import TextLogger

with open(os.path.join(os.path.dirname(__file__), '../alphastar_rl_learner_default_config.yaml')) as f:
    cfg = yaml.safe_load(f)
cfg = EasyDict(cfg)
learner_ip = '127.0.0.1'  # socket.gethostname()
coordinator_ip = '127.0.0.1'  # socket.gethostname()
cfg.api.learner_ip = learner_ip
cfg.api.coordinator_ip = coordinator_ip

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
def learner():
    learner_app = Flask("__main__")

    def run():
        try:
            learner_app.run(host=learner_ip, port=cfg.api.learner_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    logger = TextLogger(log_path, name="learner.log")
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
            coordinator_app.run(host=coordinator_ip, port=cfg.api.coordinator_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield coordinator
    # close resource operation
    coordinator.close()
