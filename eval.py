from typing import Union, Optional, List, Any, Callable, Tuple
import pickle
import torch
from functools import partial

from ding.config import compile_config, read_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed

file_name = 'collipse_recover.txt'
def eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
) -> float:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)

    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env = env_fn(evaluator_env_cfg[0])
    env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['eval']).eval_mode
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.load_state_dict(state_dict)

    obs = env.reset()
    eval_reward = 0.
    while True:
        policy_output = policy.forward({0: obs})
        logit = policy_output[0]['logit']
        action = policy_output[0]['action']
        print(logit[0])
        timestep = env.step(action)
        eval_reward += timestep.reward
        obs = timestep.obs
        if timestep.done:
            print(timestep.info)
            break

    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))


if __name__ == "__main__":
    path = '/mnt/lustre/weiyuhong/communication/DI-engine/load_model/ckpt_best.pth.tar'
    cfg = '/mnt/lustre/weiyuhong/communication/DI-engine/load_model/formatted_total_config.py'
    state_dict = torch.load(path, map_location='cpu')
    eval(cfg, seed=1, state_dict=state_dict)
