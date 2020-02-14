'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. supervised learning for Alphastar useing human replays 
'''
import sys
import os
import yaml
from easydict import EasyDict
import random
import time


from sc2learner.agents.solver import AlphastarSLLearner
from absl import flags
from absl import logging
from absl import app


FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "config.yaml", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("replay_list", None, "path to replay data")
flags.DEFINE_string("use_distributed", "False", "use distributed training")
flags.FLAGS(sys.argv)


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.data.replay_list = FLAGS.replay_list
    cfg.train.use_distributed = True if FLAGS.use_distributed == 'True' else False
    learner = AlphastarSLLearner(cfg)
    learner.run()
    learner.finalize()


if __name__ == '__main__':
    app.run(main)
