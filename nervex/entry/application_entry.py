from typing import Union, Optional, List, Any, Callable
import pickle
import torch
from functools import partial

from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.envs import create_env_manager, get_vec_env_setting
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.torch_utils import to_device
from .utils import set_pkg_seed


def eval(
        cfg: Union[str, dict],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
) -> float:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): Subclass of ``BaseEnv``, and config dict.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    cfg.policy.policy_type = cfg.policy.policy_type + '_command'
    # Env init.
    manager_cfg = cfg.env.get('manager', {})
    if env_setting is None:
        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env.env_kwargs)
    else:
        env_fn, _, evaluator_env_cfg = env_setting
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    if evaluator_env_cfg[0].get('replay_path', None):
        evaluator_env.enable_save_replay([c['replay_path'] for c in evaluator_env_cfg])
        assert cfg.env.env_manager_type == 'base'
    # Random seed.
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, cfg.policy.use_cuda)
    # Create components.
    policy = create_policy(cfg.policy, model=model, enable_field=['eval'])
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.eval_mode.load_state_dict(state_dict)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode)
    # Evaluate
    _, eval_reward = evaluator.eval()
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))

    return eval_reward


def collect_demo_data(
        cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        expert_data_path: str,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
) -> None:
    r"""
    Overview:
        Collect demostration data by the trained policy.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``Str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - collect_count (:obj:`int`): The count of collected data.
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - env_setting (:obj:`Optional[List[Any]]`): Subclass of ``BaseEnv``, and config dict.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    cfg.policy.policy_type = cfg.policy.policy_type + '_command'
    # Env init.
    if env_setting is None:
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env.env_kwargs)
    else:
        env_fn, collector_env_cfg, _ = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    # Random seed.
    collector_env.seed(seed)
    set_pkg_seed(seed, cfg.policy.use_cuda)
    # Create components.
    policy = create_policy(cfg.policy, model=model, enable_field=['collect'])
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.collect_mode.load_state_dict(state_dict)
    collector = BaseSerialCollector(cfg.collector, collector_env, policy.collect_mode)
    # let's collect some expert demostrations
    exp_data = collector.collect_data(n_sample=collect_count)
    if cfg.policy.use_cuda:
        exp_data = to_device(exp_data, 'cpu')
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)
    print('Collect demo data successfully')
