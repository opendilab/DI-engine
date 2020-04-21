import os
import sys
import time
import uuid

import numpy as np
from itertools import count
import logging
import argparse
import yaml
from easydict import EasyDict
from flask import Flask

from sc2learner.utils.log_helper import TextLogger
from .league_manager_api import create_league_manager_app
from .league_manager_wrapper import LeagueManagerWrapper

parser = argparse.ArgumentParser(description='implementation of AlphaStar')
parser.add_argument('--config', dest='config', required=True, help='settings of AlphaStar in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="league_manager.log")

cfg = yaml.safe_load(open(args.config, 'r'))
cfg = EasyDict(cfg)

league_manager_port = cfg['system']['league_manager_port']

league_manager_wrapper = LeagueManagerWrapper(cfg)
league_manager_ip = league_manager_wrapper.get_ip()

app = create_league_manager_app(league_manager_wrapper)

if __name__ == '__main__':
    app.run(host=league_manager_ip, port=league_manager_port, debug=True, use_reloader=False)
