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

from .coordinator import Coordinator
from sc2learner.utils.log_helper import TextLogger
from .coordinator_api import create_coordinator_app

parser = argparse.ArgumentParser(description='implementation of AlphaStar')
parser.add_argument('--config', dest='config', required=True, help='settings of AlphaStar in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="coordinator.log")

cfg = yaml.safe_load(open(args.config, 'r'))
cfg = EasyDict(cfg)
cfg.system.resume_dir = log_path
coordinator_ip = cfg['system']['coordinator_ip']
coordinator_port = cfg['system']['coordinator_port']

coordinator = Coordinator(cfg)
app = create_coordinator_app(coordinator)

if __name__ == '__main__':
    app.run(host=coordinator_ip, port=coordinator_port, debug=True, use_reloader=False)
