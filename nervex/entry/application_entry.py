from typing import Union, Optional, List, Any
import pickle
import torch
from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommander
from nervex.worker import BaseEnvManager, SubprocessEnvManager
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting
from nervex.torch_utils import to_device
from .utils import set_pkg_seed


def eval(
        cfg: Union[str, dict],
        seed: int,
        env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
        policy_type: Optional[type] = None,  # subclass of Policy
        model: Optional[Union[type, torch.nn.Module]] = None,  # instance or subclass of torch.nn.Module
        state_dict: Optional[dict] = None,  # policy or model state_dict
) -> None:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``Str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[Any]`): Subclass of ``BaseEnv``, and config dict.
        - policy_type (:obj:`Optional[type]`): Subclass of ``Policy``.
        - model (:obj:`Optional[Union[type, torch.nn.Module]]`): Instance or subclass of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    # Env init.
    manager_cfg = cfg.env.get('manager', {})
    if env_setting is None:
        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, _, evaluator_env_cfg = env_setting
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    evaluator_env = env_manager_type(
        env_fn,
        env_cfg=evaluator_env_cfg,
        env_num=len(evaluator_env_cfg),
        manager_cfg=manager_cfg,
        episode_num=manager_cfg.get('episode_num', len(evaluator_env_cfg))
    )
    if evaluator_env_cfg[0].get('replay_path', None):
        evaluator_env.enable_save_replay([c['replay_path'] for c in evaluator_env_cfg])
        assert cfg.env.env_manager_type == 'base'
    # Random seed.
    evaluator_env.seed(seed)
    set_pkg_seed(seed, cfg.policy.use_cuda)
    # Create components.
    policy_fn = create_policy if policy_type is None else policy_type
    policy = policy_fn(cfg.policy, model=model, enable_field=['eval'])
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.state_dict_handle()['model'].load_state_dict(state_dict['model'])
    evaluator = BaseSerialEvaluator(cfg.evaluator)

    evaluator.env = evaluator_env
    evaluator.policy = policy.eval_mode
    # Evaluate
    _, eval_reward = evaluator.eval(0)
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))
    evaluator.close()


def collect_demo_data(
        cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        expert_data_path: str,
        env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
        policy_type: Optional[type] = None,  # subclass of Policy
        model: Optional[Union[type, torch.nn.Module]] = None,  # instance or subclass of torch.nn.Module
        state_dict: Optional[dict] = None,  # policy or model state_dict
) -> None:
    r"""
    Overview:
        Collect demostration data by the trained policy.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``Str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - collect_count (:obj:`int`): The count of collected data.
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - env_setting (:obj:`Optional[Any]`): Subclass of ``BaseEnv``, and config dict.
        - policy_type (:obj:`Optional[type]`): Subclass of ``Policy``.
        - model (:obj:`Optional[Union[type, torch.nn.Module]]`): Instance or subclass of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    # Env init.
    manager_cfg = cfg.env.get('manager', {})
    if env_setting is None:
        env_fn, actor_env_cfg, _ = get_vec_env_setting(cfg.env)
    else:
        env_fn, actor_env_cfg, _ = env_setting
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    actor_env = env_manager_type(
        env_fn,
        env_cfg=actor_env_cfg,
        env_num=len(actor_env_cfg),
        manager_cfg=manager_cfg,
    )
    if actor_env_cfg[0].get('replay_path', None):
        actor_env.enable_save_replay([c['replay_path'] for c in actor_env_cfg])
        assert cfg.env.env_manager_type == 'base'
    # Random seed.
    actor_env.seed(seed)
    set_pkg_seed(seed, cfg.policy.use_cuda)
    # Create components.
    policy_fn = create_policy if policy_type is None else policy_type
    policy = policy_fn(cfg.policy, model=model, enable_field=['collect'])
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.state_dict_handle()['model'].load_state_dict(state_dict['model'])
    actor = BaseSerialActor(cfg.actor)

    actor.env = actor_env
    actor.policy = policy.collect_mode
    # let's collect some expert demostrations
    exp_data = actor.generate_data(n_sample=collect_count, iter_count=-1)
    if cfg.policy.use_cuda:
        exp_data = to_device(exp_data, 'cpu')
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)
    print('Collect demo data successfully')
    actor.close()
