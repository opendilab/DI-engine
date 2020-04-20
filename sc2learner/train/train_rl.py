'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. reinfocement learning training for Alphastar(league training)
'''
import os
import sys

from absl import app, flags, logging

from sc2learner.utils import read_config, merge_dicts
from sc2learner.worker.learner import AlphaStarRLLearner
from sc2learner.worker.actor import AlphaStarActorWorker

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("job_name", "", "job name:[learner, actor]")
flags.FLAGS(sys.argv)

# default_config = read_config(os.path.join(os.path.dirname(__file__), "train_sl_default_config.yaml"))


def main(argv):
    logging.set_verbosity(logging.ERROR)
    # cfg = merge_dicts(default_config, read_config(FLAGS.config_path))
    cfg = read_config(FLAGS.config_path)

    # Fill config items
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path

    # Start job
    assert FLAGS.job_name in ['learner', 'actor']
    if FLAGS.job_name == 'learner':
        learner = AlphaStarRLLearner(cfg)
        learner.run()
        learner.finalize()
    elif FLAGS.job_name == 'actor':
        actor = AlphaStarActorWorker(cfg)
        actor.run()


if __name__ == '__main__':
    app.run(main)
