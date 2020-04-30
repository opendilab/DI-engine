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
from sc2learner.worker.learner import create_learner_app
from sc2learner.worker.actor import AlphaStarActorWorker

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("job_name", "", "job name:[learner, actor]")
flags.FLAGS(sys.argv)


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
        # learner.run()
        # learner.finalize()
        learner_app = create_learner_app(learner)
        learner_ip, learner_port = learner.get_ip_port()
        learner_app.run(host=learner_ip, port=learner_port, debug=True, use_reloader=False)
    elif FLAGS.job_name == 'actor':
        actor = AlphaStarActorWorker(cfg)
        actor.run()


if __name__ == '__main__':
    app.run(main)
