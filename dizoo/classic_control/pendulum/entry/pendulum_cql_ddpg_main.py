import torch
from copy import deepcopy

from dizoo.classic_control.pendulum.config.pendulum_ddpg_data_generation_config import main_config, create_config
from ding.entry import serial_pipeline_offline, collect_demo_data, eval, serial_pipeline


def train_cql(args):
    from dizoo.classic_control.pendulum.config.pendulum_cql_config import main_config, create_config
    main_config.exp_name = 'cql_ddpg'
    main_config.policy.learn.data_path = './ddpg/expert_demos.hdf5'
    main_config.policy.learn.data_type = 'hdf5'
    config = deepcopy([main_config, create_config])
    serial_pipeline_offline(config, seed=args.seed)


def eval_ckpt(args):
    main_config.exp_name = 'ddpg'
    main_config.policy.learn.learner.load_path = './ddpg/ckpt/ckpt_best.pth.tar'
    main_config.policy.learn.learner.hook.load_ckpt_before_run = './ddpg/ckpt/ckpt_best.pth.tar'
    config = deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path=main_config.policy.learn.learner.hook.load_ckpt_before_run)


def generate(args):
    main_config.exp_name = 'ddpg'
    main_config.policy.learn.learner.load_path = './ddpg/ckpt/ckpt_best.pth.tar'
    main_config.policy.learn.save_path = './ddpg/expert.pkl'
    # main_config.policy.learn.data_type = 'hdf5'
    config = deepcopy([main_config, create_config])
    state_dict = torch.load(main_config.policy.learn.learner.load_path, map_location='cpu')
    collect_demo_data(
        config,
        collect_count=main_config.policy.other.replay_buffer.replay_buffer_size,
        seed=args.seed,
        expert_data_path=main_config.policy.learn.save_path,
        state_dict=state_dict
    )


def train_expert(args):
    from dizoo.classic_control.pendulum.config.pendulum_ddpg_config import main_config, create_config
    main_config.exp_name = 'ddpg'
    config = deepcopy([main_config, create_config])
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    args = parser.parse_args()

    # train_expert(args)
    # eval_ckpt(args)
    generate(args)
    # train_cql(args)
