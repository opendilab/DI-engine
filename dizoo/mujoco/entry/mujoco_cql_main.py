from ding.config import read_config
from dizoo.mujoco.config import hopper_cql_default_config
from ding.entry import serial_pipeline_offline

def train(args):
    config = read_config(hopper_cql_default_config)
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    args = parser.parse_args()

    train(args)
