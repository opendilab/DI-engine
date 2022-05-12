from ding.entry import serial_pipeline_bc, serial_pipeline, collect_demo_data
# from dizoo.box2d.bipedalwalker.config.bipedalwalker_sac_config import main_config, create_config
from dizoo.classic_control.pendulum.config.pendulum_td3_config import main_config, create_config
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
    """
    Overview:
        Serial pipeline entry for off-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    sd = torch.load(load_path, map_location='cpu')
    policy.collect_mode.load_state_dict(sd)
    return policy


def run_go():
    bipe_sac_config, bipe_sac_create_config = main_config, create_config
    train_config = [deepcopy(bipe_sac_config), deepcopy(bipe_sac_create_config)]
    exp_path = '/mnt/nfs/wanghaolin/cont_bc/DI-engine/pendulum_td3_seed0/ckpt/ckpt_best.pth.tar'
    expert_policy = load_policy(train_config, load_path=exp_path, seed=0)

    # collect expert demo data
    collect_count = 400
    expert_data_path = 'expert_data_dqn.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(bipe_sac_config), deepcopy(bipe_sac_create_config)]
    # collect_demo_data(
    #     collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    # )
    collect_episodic_demo_data(
        deepcopy(collect_config), seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )
    collect_episodic_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path='eval_data.pkl', collect_count=10,
    )

    episode_to_transitions(
        expert_data_path, expert_data_path, nstep=1
    )

    # il training 2
    il_config = [deepcopy(bipe_sac_config), deepcopy(bipe_sac_create_config)]
    il_config[0].policy.learn.train_epoch = 100
    il_config[0].policy.type = 'continuous_bc'
    il_config[0].exp_name = "continuous_bc_seed0"
    il_config[0].env.stop_value = 50
    il_config[0].multi_agent = False
    bc_policy , converge_stop_flag = serial_pipeline_bc(il_config, seed=314, data_path=expert_data_path)
    return bc_policy


policy = run_go()
with open('eval_data.pkl', 'rb') as f:
    eval_data = pickle.load(f)

for epi in eval_data:
    loss_tot = []
    for ite in epi:
        pred = policy._model(ite['obs'].cuda(), mode='compute_actor')['action']
        real = ite['action']
        loss_tot.append(nn.L1Loss()(pred, real).item())
    print(loss_tot)
