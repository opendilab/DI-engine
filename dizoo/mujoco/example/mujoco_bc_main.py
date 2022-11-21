from ding.entry import serial_pipeline_bc, serial_pipeline, collect_demo_data
from dizoo.mujoco.config.halfcheetah_td3_config import main_config, create_config
from copy import deepcopy
from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import torch.nn as nn
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.entry.utils import random_collect
from ding.entry import collect_demo_data, collect_episodic_demo_data, episode_to_transitions
import pickle


def load_policy(
        input_cfg: Union[str, Tuple[dict, dict]],
        load_path: str,
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
) -> 'Policy':  # noqa
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    sd = torch.load(load_path, map_location='cpu')
    policy.collect_mode.load_state_dict(sd)
    return policy


def main():
    half_td3_config, half_td3_create_config = main_config, create_config
    train_config = [deepcopy(half_td3_config), deepcopy(half_td3_create_config)]
    exp_path = 'DI-engine/halfcheetah_td3_seed0/ckpt/ckpt_best.pth.tar'
    expert_policy = load_policy(train_config, load_path=exp_path, seed=0)

    # collect expert demo data
    collect_count = 100
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(half_td3_config), deepcopy(half_td3_create_config)]

    collect_episodic_demo_data(
        deepcopy(collect_config),
        seed=0,
        state_dict=state_dict,
        expert_data_path=expert_data_path,
        collect_count=collect_count
    )

    episode_to_transitions(expert_data_path, expert_data_path, nstep=1)

    # il training 2
    il_config = [deepcopy(half_td3_config), deepcopy(half_td3_create_config)]
    il_config[0].policy.learn.train_epoch = 1000000
    il_config[0].policy.type = 'bc'
    il_config[0].policy.continuous = True
    il_config[0].exp_name = "continuous_bc_seed0"
    il_config[0].env.stop_value = 50000
    il_config[0].multi_agent = False
    bc_policy, converge_stop_flag = serial_pipeline_bc(il_config, seed=314, data_path=expert_data_path, max_iter=4e6)
    return bc_policy


if __name__ == '__main__':
    policy = main()
