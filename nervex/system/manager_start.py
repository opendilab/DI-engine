import os
import sys
import time
import uuid

import numpy as np
import logging
import argparse

from .manager import Manager
from nervex.utils.log_helper import TextLogger
from nervex.utils import read_config
from .manager_api import create_manager_app

parser = argparse.ArgumentParser(description='nervex manager(forward communication message) start point')
parser.add_argument('--config', dest='config', required=True, help='manager settings in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="manager.log")

cfg = read_config(args.config)
cfg.system.resume_dir = log_path
manager_ip = cfg.system.manager_ip
manager_port = cfg.system.manager_port

manager = Manager(cfg)
app = create_manager_app(manager)

if __name__ == '__main__':
    app.run(host=manager_ip, port=manager_port, debug=True, use_reloader=False)
