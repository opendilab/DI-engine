from __future__ import division
import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os
import time
import copy
from itertools import count
import multiprocessing
import argparse
import yaml
import json
import math
import logging
import socket
from flask import Flask, request

from utils.log_helper import TextLogger

from fake_learner import FakeLearner


parser = argparse.ArgumentParser(description='implementation of AlphaStar')
parser.add_argument('--config', dest='config', required=True, help='settings of AlphaStar in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="learner.log")

cfg = yaml.load(open(args.config, 'r'))
api_info = cfg['api']
learner_ip = socket.gethostname()
learner_port = api_info['learner_port']
coordinator_ip = api_info['coordinator_ip']
coordinator_port = api_info['coordinator_port']
manager_ip = api_info['manager_ip']
manager_port = api_info['manager_port']

app = Flask(__name__)

# # todo:
# learner = learner_entry("alphastar_agent", cfg) 
learner = FakeLearner(cfg)

def build_ret(code, info=''):
    return {'code': code, 'info': info}


@app.route('/learner/get_metadata', methods=['POST'])
def get_sample():
    metadata = request.json['metadata']
    info = learner.deal_with_get_metadata(metadata)
    return build_ret(0, info)


###################################################################################
#                                      debug                                      #
###################################################################################



if __name__ == '__main__':
    app.run(host=learner_ip, port=learner_port, debug=True, use_reloader=False)

