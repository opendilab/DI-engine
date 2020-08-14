import os
import sys
import time
import uuid

import numpy as np
from itertools import count
import logging
import argparse
import yaml

from manager import Manager
from nervex.utils.log_helper import TextLogger
from manager_api import create_manager_app

parser = argparse.ArgumentParser(description='implementation of AlphaStar')
parser.add_argument('--config', dest='config', required=True, help='settings of AlphaStar in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="manager.log")

cfg = yaml.safe_load(open(args.config, 'r'))
# TODO: separate modules
api_info = cfg['api']
manager_ip = api_info['manager_ip']
manager_port = api_info['manager_port']

manager = Manager(cfg)
app = create_manager_app(manager)

if __name__ == '__main__':
    app.run(host=manager_ip, port=manager_port, debug=True, use_reloader=False)
