import pytest
import copy
import os
import sys
import socket
from threading import Thread
from nervex.system import Coordinator, create_coordinator_app, LeagueManagerWrapper, create_league_manager_app
from nervex.utils.log_helper import TextLogger
from nervex.utils import read_config

log_path = os.path.join(os.path.dirname(__file__))
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
if not os.path.exists(log_path):
    os.mkdir(log_path)


def build_ret(code, info=''):
    return {'code': code, 'info': info}


@pytest.fixture(scope='class')
def setup_config():
    path = os.path.join(os.path.dirname(__file__), 'test_dist.yaml')
    cfg = read_config(path)
    return cfg


@pytest.fixture(scope='class')
def coordinator(setup_config):
    cfg = setup_config
    logger = TextLogger(log_path, name="coordinator.log")
    coordinator = Coordinator(cfg)
    coordinator_app = create_coordinator_app(coordinator)

    def run():
        try:
            coordinator_app.run(
                host=cfg.system.coordinator_ip, port=cfg.system.coordinator_port, debug=True, use_reloader=False
            )
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield coordinator
    # close resource operation
    coordinator.close()


@pytest.fixture(scope='class')
def league_manager(setup_config):
    cfg = setup_config
    logger = TextLogger(log_path, name="league_manager.log")
    league_manager_wrapper = LeagueManagerWrapper(cfg)
    app = create_league_manager_app(league_manager_wrapper)

    def run():
        try:
            app.run(
                host=cfg.system.league_manager_ip, port=cfg.system.league_manager_port, debug=True, use_reloader=False
            )
        except KeyboardInterrupt:
            pass

    launch_thread = Thread(target=run)
    launch_thread.daemon = True
    launch_thread.start()
    yield league_manager_wrapper
    # close resource operation
    league_manager_wrapper.close()
