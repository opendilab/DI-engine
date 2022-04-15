from dizoo.d4rl.config.hopper_td3bc_medium_expert_config import main_config, create_config
from ding.entry import serial_pipeline_offline
import os



def train(args):
    config = [main_config, create_config]
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    from d4rl import set_dataset_path
    
    # set environment variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    args = parser.parse_args()
    train(args)
