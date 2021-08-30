from dizoo.mujoco.config.hopper_sac_data_generation_default_config import main_config, create_config
from ding.entry import serial_pipeline_data_generation


def generate(args):
    config = [main_config, create_config]
    serial_pipeline_data_generation(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    generate(args)
