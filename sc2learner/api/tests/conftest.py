import pytest
import os
import sys
import yaml
import socket
import signal
from easydict import EasyDict
from threading import Thread
from flask import Flask, request
from ..learner_communication_helper import LearnerCommunicationHelper
from ..coordinator import Coordinator
from sc2learner.utils.log_helper import TextLogger

with open(os.path.join(os.path.dirname(__file__), '../config.yaml')) as f:
    cfg = yaml.safe_load(f)
cfg = EasyDict(cfg)
learner_ip = socket.gethostname()
coordinator_ip = socket.gethostname()
cfg.api.learner_ip = learner_ip
cfg.api.coordinator_ip = coordinator_ip
cfg.data = {}
cfg.data.train = {}
cfg.data.train.batch_size = 4

log_path = os.path.join(os.path.dirname(__file__))
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
if not os.path.exists(log_path):
    os.mkdir(log_path)


def build_ret(code, info=''):
    return {'code': code, 'info': info}


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
    coordinator_app = Flask("__main__")

    def run():
        try:
            coordinator_app.run(host=coordinator_ip, port=cfg.api.coordinator_port, debug=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    @coordinator_app.route('/coordinator/register_learner', methods=['POST'])
    def register_learner():
        learner_uid = request.json['learner_uid']
        learner_ip = request.json['learner_ip']
        ret_code = coordinator.deal_with_register_learner(learner_uid, learner_ip)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @coordinator_app.route('/coordinator/ask_for_metadata', methods=['POST'])
    def ask_for_metadata():
        learner_uid = request.json['learner_uid']
        batch_size = request.json['batch_size']
        ret = coordinator.deal_with_ask_for_metadata(learner_uid, batch_size)
        if ret:
            return build_ret(0, ret)
        else:
            return build_ret(1)

    logger = TextLogger(log_path, name="coordinator.log")
    coordinator = Coordinator(cfg)
    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield coordinator
    # close resource operation
    coordinator.close()
