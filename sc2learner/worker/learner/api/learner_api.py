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

from sc2learner.utils.log_helper import TextLogger
# from fake_learner import FakeLearner
from sc2learner.utils import read_config, merge_dicts
from sc2learner.worker.learner import AlphaStarRLLearner
from sc2learner.worker.actor import AlphaStarActorWorker


def create_learner_app(learner):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/learner/reset', methods=['POST'])
    def reset():
        checkpoint_path = request.json['checkpoint_path']
        info = learner.reset(checkpoint_path)
        return build_ret(0, info)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    @app.route('/learner/get_trajectory_record_len', methods=['GET'])
    def get_trajectory_record_len():
        info = learner.deal_with_get_trajectory_record_len()
        return build_ret(0, info)

    @app.route('/learner/set_delete_trajectory_record', methods=['POST'])
    def set_delete_trajectory_record():
        state = request.json['state']
        num = request.json['num']
        info = learner.deal_with_set_delete_trajectory_record(state, num)
        return build_ret(0, info)

    @app.route('/learner/set_train_freq', methods=['POST'])
    def set_train_freq():
        freq = request.json['freq']
        info = learner.deal_with_set_train_freq(freq)
        return build_ret(0, info)

    return app
