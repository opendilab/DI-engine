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
flags.FLAGS(sys.argv)


def main(argv):
    logging.set_verbosity(logging.ERROR)
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.data.replay_list = FLAGS.replay_list
    learner = AlphastarSLLearner(cfg)
    learner.run()


if __name__ == '__main__':
    app.run(main)
