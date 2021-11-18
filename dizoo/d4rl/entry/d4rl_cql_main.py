from dizoo.d4rl.config.hopper_expert_cql_default_config import main_config, create_config
from ding.entry import serial_pipeline_offline


def train(args):
    config = [main_config, create_config]
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    args = parser.parse_args()

    train(args)
