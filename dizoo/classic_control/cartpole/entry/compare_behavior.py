from typing import Union, Optional, List, Any, Tuple
import pickle
import torch
from functools import partial

from ding.config import compile_config, read_config
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.torch_utils import to_device
from ding.utils import set_pkg_seed
from ding.utils.data import offline_data_save_type
from ding.reward_model import create_reward_model
from dizoo.classic_control.cartpole.config.cartpole_gail_config import cartpole_gail_config, cartpole_gail_create_config
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from tensorboardX import SummaryWriter
import os
import numpy as np
from scipy.stats import wasserstein_distance


def eval_behavior(
        cfg: Union[str, Tuple[dict, dict]],
        exp_cfg: Union[str, Tuple[dict, dict]],
        actions_type: str,
        seed: int = 0,
        cut_expert_rewards: Optional[int] = None,
) -> float:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
        - load_path (:obj:`Optional[str]`): Path to load ckpt.
    """
    assert actions_type in ['discrete', 'continuous']
    if isinstance(cfg, str):
        cfg, create_cfg = read_config(cfg)
    else:
        cfg, create_cfg = cfg
    if isinstance(exp_cfg, str):
        exp_cfg, exp_create_cfg = read_config(exp_cfg)
    else:
        exp_cfg, exp_create_cfg = exp_cfg
    cfg = compile_config(
        cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=False, save_path='eval_config.py'
    )
    exp_cfg = compile_config(
        exp_cfg, seed=seed, auto=True, create_cfg=exp_create_cfg, save_cfg=False, save_path='exp_eval_config.py'
    )
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, enable_field=['eval'])
    policy = policy.eval_mode
    policy.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))
    exp_policy = create_policy(exp_cfg.policy, enable_field=['eval'])
    exp_policy = exp_policy.eval_mode
    exp_policy.load_state_dict(torch.load(exp_cfg.policy.load_path, map_location='cpu'))

    with open(cfg.reward_model.expert_data_path, 'rb') as f:
        expert_data_loader: list = pickle.load(f)
    k = min(len(expert_data_loader), cut_expert_rewards) if cut_expert_rewards is not None else len(expert_data_loader)
    expert_data_loader = expert_data_loader[:k]
    tot_tv = []  # https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-total-variation-distance.html
    divergent_actions = []
    w_dist = []  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    l1_dist = []
    for item in expert_data_loader:
        data = {0: item['obs']}
        log = policy.forward(data)
        exp_log = exp_policy.forward(data)
        if actions_type == 'discrete':
            p = torch.softmax(log[0]['logit'], dim=-1).detach().cpu().numpy()
            exp_p = torch.softmax(exp_log[0]['logit'], dim=-1).detach().cpu().numpy()
            a, exp_a = np.argmax(p), np.argmax(exp_p)
            divergent_actions.append(a != exp_a)
            diff = np.absolute(p - exp_p)
            tot_tv.append(0.5 * np.sum(diff))
            w_dist.append(wasserstein_distance(p, exp_p))
        if actions_type == 'continuous':
            a = torch.softmax(log[0]['action'], dim=-1).detach().cpu().numpy()
            exp_a = torch.softmax(exp_log[0]['action'], dim=-1).detach().cpu().numpy()
            diff = np.absolute(a - exp_a)
            l1_dist.append(np.sum(diff))
            w_dist.append(wasserstein_distance(a, exp_a))
    if actions_type == 'discrete':
        divergent_actions = (sum(divergent_actions) / len(divergent_actions)) * 100
        print('Divergent actions:', divergent_actions, '%')
        avg_tv = np.average(tot_tv)
        print('TV distance:', avg_tv)
    if actions_type == 'continuous':
        l1_dist = np.average(l1_dist)
        print('L1 distance:', l1_dist)
    avg_w = np.average(w_dist)
    print('Wasserstein distance:', avg_w)

if __name__ == "__main__":
    eval_behavior((cartpole_gail_config, cartpole_gail_create_config),
                  (cartpole_dqn_config, cartpole_dqn_create_config),
                  actions_type='discrete',
                  seed=0, cut_expert_rewards=10000)
