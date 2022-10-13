# Code Reference: https://github.com/OptMLGroup/DeepBeerInventory-RL.import numpy as np
import pickle
import json
import random
import matplotlib
if not True:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

import sys


class Logger(object):
    # writes the outputs to a file
    def __init__(self, config):
        self.terminal = sys.stdout
        self.log = open(os.path.join(config.model_dir, "logfile.log"), "a", 0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def prepare_dirs_and_logger(config):
    # set format of the logger
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger('tensorflow')

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # create load paths, if they don't exist
    if config.load_path:
        if config.load_path.startswith(config.task):
            config.model_name = config.load_path
        else:
            config.model_name = "{}_{}".format(config.task, config.load_path)
    else:
        if config.iftl:
            tl = 1
        else:
            tl = 0
        config.model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            config.task, get_time(), config.gameConfig, config.tlBaseBrain, config.NoHiLayer, config.demandUp,
            config.cp1, config.cp2, config.cp3, config.cp4, config.ch1, config.ch2, config.ch3, config.ch4,
            config.distCoeff, config.NoAgent, config.maxEpisodesTrain, config.lr0, config.multPerdInpt, config.dnnUpCnt,
            tl, config.actionUp, config.demandDistribution, config.action_step, config.data_id
        )

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.pre_model_dir, config.log_dir, config.model_dir, os.path.join(config.model_dir,
                                                                                      'saved_figures'),
                 os.path.join(config.model_dir, 'model1'), os.path.join(config.model_dir, 'model2'),
                 os.path.join(config.model_dir, 'model3'), os.path.join(config.model_dir, 'model4'),
                 os.path.join(config.pre_model_dir, 'brain3'), os.path.join(config.pre_model_dir, 'brain4'),
                 os.path.join(config.pre_model_dir,
                              'brain5'), os.path.join(config.pre_model_dir,
                                                      'brain6'), os.path.join(config.pre_model_dir, 'brain7'),
                 os.path.join(config.pre_model_dir,
                              'brain8'), os.path.join(config.pre_model_dir,
                                                      'brain9'), os.path.join(config.pre_model_dir,
                                                                              'brain10'), os.path.join(config.log_dir,
                                                                                                       'reports')]:

        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# input-output functions
def save(obj, name):
    pickle.dump(obj, open(name, "wb"))


def load(name):
    return pickle.load(open(name, "rb"))
