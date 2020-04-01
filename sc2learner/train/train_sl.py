'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. supervised learning for Alphastar using human replays
'''
import os
import sys

from absl import app, flags, logging

from sc2learner.utils import read_config, merge_dicts
from sc2learner.worker.learner.alphastar_sl_learner import AlphaStarSupervisedLearner

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "", "path to config file")
flags.DEFINE_string("load_path", "", "path to model checkpoint")
flags.DEFINE_string("replay_list", None, "path to replay data")
flags.DEFINE_string("eval_replay_list", None, "path to evaluate replay data")
flags.DEFINE_bool("use_distributed", False, "use distributed training")
flags.DEFINE_bool("only_evaluate", False, "only sl evaluate")
flags.DEFINE_bool("use_fake_dataset", False, "whether to use fake dataset")
flags.FLAGS(sys.argv)

default_config = read_config(os.path.join(os.path.dirname(__file__), "train_default_config.yaml"))


def main(argv):
    logging.set_verbosity(logging.ERROR)
    cfg = merge_dicts(default_config, read_config(FLAGS.config_path))

    # Fill config items
    cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    cfg.common.load_path = FLAGS.load_path
    cfg.common.only_evaluate = FLAGS.only_evaluate
    cfg.data.train.replay_list = FLAGS.replay_list
    cfg.data.eval.replay_list = FLAGS.eval_replay_list
    cfg.train.use_distributed = FLAGS.use_distributed
    cfg.data.train.use_distributed = FLAGS.use_distributed
    cfg.data.eval.use_distributed = FLAGS.use_distributed
    if FLAGS.use_fake_dataset:
        cfg.data.train.dataset_type = "fake"
        cfg.data.eval.dataset_type = "fake"

    # Start learning
    learner = AlphaStarSupervisedLearner(cfg)
    learner.run()
    learner.finalize()


if __name__ == '__main__':
    app.run(main)
