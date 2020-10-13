import argparse

from nervex.utils import read_config
from nervex.worker.learner import create_learner


def main(cfg):
    learner = create_learner(cfg)
    learner.run()
    learner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nervex actor start point")
    parser.add_argument("--config", dest="config", required=True, help="actor settings in yaml format")
    args = parser.parse_args()
    cfg = read_config(args.config)
    main(cfg)
