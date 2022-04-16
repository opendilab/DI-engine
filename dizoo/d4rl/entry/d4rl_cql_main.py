from ding.entry import serial_pipeline_offline
from ding.config import read_config
import os


def train(args):
    if '../config' in args.config:
        config = read_config(args.config)
    else:
        config = read_config(os.path.join('../config/', args.config))
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    from d4rl import set_dataset_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--config', '-c', type=str, default='hopper_expert_cql_config.py')
    args = parser.parse_args()
    train(args)
