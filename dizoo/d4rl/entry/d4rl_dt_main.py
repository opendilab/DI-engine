from ding.entry import serial_pipeline_dt
from ding.config import read_config
import os
from copy import deepcopy


def train(args):
    config_dir = '/mnt/lustre/wangzilin/research/dt/DI-engine/dizoo/d4rl/config'
    if config_dir in args.config:
        main_config, create_config = read_config(args.config)
    else:
        main_config, create_config = read_config(os.path.join(config_dir, args.config))

    result_root = '/mnt/lustre/wangzilin/research/dt/dt/v2'

    itms = args.config.split('_')
    env = itms[0]
    sub_env = '_'.join(itms[1:3])
    if '_dt' in sub_env:
        sub_env = sub_env[:-3]
    print(env, sub_env)

    for seed in [0, 1, 2]:

        args.seed = seed

        log_dir = '_'.join([env, sub_env, str(args.seed)])

        main_config.policy.log_dir = os.path.join(result_root, env, sub_env, log_dir)

        exp_name = '_'.join([env, sub_env, 'seed', str(args.seed)])

        main_config.exp_name = exp_name
        config = deepcopy([main_config, create_config])
        serial_pipeline_dt(config, seed=args.seed, max_train_iter=3000)


if __name__ == "__main__":
    import argparse
    from d4rl import set_dataset_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--config', '-c', type=str, default='hopper_medium_expert_dt_config.py')
    args = parser.parse_args()
    train(args)
