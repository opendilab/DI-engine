from dizoo.mujoco.config.hopper_sac_data_generation_default_config import main_config, create_config
from ding.entry import collect_demo_data, eval
import torch
import copy


def eval_ckpt(args):
    config = copy.deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path=main_config.policy.learn.learner.hook.load_ckpt_before_run)


def generate(args):
    config = copy.deepcopy([main_config, create_config])
    state_dict = torch.load(main_config.policy.learn.learner.load_path, map_location='cpu')
    collect_demo_data(
        config,
        collect_count=main_config.policy.other.replay_buffer.replay_buffer_size,
        seed=args.seed,
        expert_data_path=main_config.policy.collect.save_path,
        state_dict=state_dict
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    eval_ckpt(args)
    generate(args)
