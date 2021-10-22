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
from dizoo.box2d.lunarlander.config.lunarlander_gail_config import lunarlander_gail_default_config,\
    lunarlander_gail_create_config
from tensorboardX import SummaryWriter
import os


def eval_reward(
        rew_model_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        load_path: Optional[str] = None,
        cut_expert_rewards: Optional[int] = None
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
    if isinstance(rew_model_cfg, str):
        cfg, create_cfg = read_config(rew_model_cfg)
    else:
        cfg, create_cfg = rew_model_cfg
    cfg = compile_config(
        cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=False, save_path='eval_config.py'
    )
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    reward_model = create_reward_model(cfg.reward_model, 'cpu', None)
    if state_dict is None:
        if load_path is None:
            load_path = cfg.policy.load_path
        state_dict = torch.load(load_path, map_location='cpu')
    reward_model.load_state_dict(state_dict)

    with open(cfg.reward_model.expert_data_path, 'rb') as f:
        expert_data_loader: list = pickle.load(f)
    exp_reward = []
    k = min(len(expert_data_loader), cut_expert_rewards) if cut_expert_rewards is not None else len(expert_data_loader)
    expert_data_loader = expert_data_loader[:k]
    for item in expert_data_loader:
        exp_reward.append(item['reward'])
    data = reward_model.estimate(expert_data_loader)
    model_reward = []
    for item in expert_data_loader:
        model_reward.append(item['reward'])

    tb_logger_exp = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'reward_expert'))
    tb_logger_mod = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'reward_model'))
    for i in range(len(model_reward)):
        # print(exp_reward[i], model_reward[i])
        tb_logger_exp.add_scalar('reward', exp_reward[i], i)
        tb_logger_mod.add_scalar('reward', model_reward[i], i)
    tb_logger_exp.close()
    tb_logger_mod.close()


if __name__ == "__main__":
    eval_reward((lunarlander_gail_default_config, lunarlander_gail_create_config), seed=0, cut_expert_rewards=10000)
