from ding.entry import serial_pipeline_offline
from ding.config import read_config
from pathlib import Path


def train(args):
    # launch from anywhere
    config = Path(__file__).absolute().parent.parent / 'config' / args.config 
    config = read_config(str(config))
    config[0].exp_name = config[0].exp_name.replace('0', str(args.seed))
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--config', '-c', type=str, default='hopper_expert_cql_config.py')
    args = parser.parse_args()
    train(args)
