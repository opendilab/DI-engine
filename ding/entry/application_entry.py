from typing import Union, Optional, List, Any, Tuple
import os
import pickle
import numpy as np
import torch
from functools import partial
from copy import deepcopy

from ding.config import compile_config, read_config
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, EpisodeSerialCollector
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.torch_utils import to_device, to_ndarray
from ding.utils import set_pkg_seed
from ding.utils.data import offline_data_save_type
from ding.rl_utils import get_nstep_return_data
from ding.utils.data import default_collate


def eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        load_path: Optional[str] = None,
        replay_path: Optional[str] = None,
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
        - load_path (:obj:`Optional[str]`): Path to load ckpt.
        - replay_path (:obj:`Optional[str]`): Path to save replay.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(
        cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True, save_path='eval_config.py'
    )

    # Create components: env, policy, evaluator
    if env_setting is None:
        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    else:
        env_fn, _, evaluator_env_cfg = env_setting
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator_env.seed(seed, dynamic_seed=False)
    if replay_path is None:  # argument > config
        replay_path = cfg.env.get('replay_path', None)
    if replay_path:
        evaluator_env.enable_save_replay(replay_path)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['eval'])
    if state_dict is None:
        if load_path is None:
            load_path = cfg.policy.learn.learner.load_path
        state_dict = torch.load(load_path, map_location='cpu')
    policy.eval_mode.load_state_dict(state_dict)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode)

    # Evaluate
    _, episode_info = evaluator.eval()
    reward = [e['final_eval_reward'] for e in episode_info]
    eval_reward = np.mean(to_ndarray(reward))
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))
    return eval_reward


def collect_demo_data(
        input_cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        expert_data_path: Optional[str] = None,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        state_dict_path: Optional[str] = None,
) -> None:
    r"""
    Overview:
        Collect demonstration data by the trained policy.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - collect_count (:obj:`int`): The count of collected data.
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
        - state_dict_path (:obj:`Optional[str]`): The path of the state_dict of policy or model.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(
        cfg,
        seed=seed,
        env=env_fn,
        auto=True,
        create_cfg=create_cfg,
        save_cfg=True,
        save_path='collect_demo_data_config.py'
    )
    if expert_data_path is None:
        expert_data_path = cfg.policy.collect.save_path

    # Create components: env, policy, collector
    if env_setting is None:
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env, eval_=False)
    else:
        env_fn, collector_env_cfg, _ = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['collect', 'eval'])
    # for policies like DQN (in collect_mode has eps-greedy)
    # collect_demo_policy = policy.collect_function(
    #     policy._forward_eval,
    #     policy._process_transition,
    #     policy._get_train_sample,
    #     policy._reset_eval,
    #     policy._get_attribute,
    #     policy._set_attribute,
    #     policy._state_dict_collect,
    #     policy._load_state_dict_collect,
    # )
    collect_demo_policy = policy.collect_mode
    if state_dict is None:
        assert state_dict_path is not None
        state_dict = torch.load(state_dict_path, map_location='cpu')
    policy.collect_mode.load_state_dict(state_dict)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, collect_demo_policy)

    if hasattr(cfg.policy.other, 'eps'):
        policy_kwargs = {'eps': 0.}
    else:
        policy_kwargs = None

    # Let's collect some expert demonstrations
    exp_data = collector.collect(n_sample=collect_count, policy_kwargs=policy_kwargs)
    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    # Save data transitions.
    offline_data_save_type(exp_data, expert_data_path, data_type=cfg.policy.collect.get('data_type', 'naive'))
    print('Collect demo data successfully')


def collect_episodic_demo_data(
        input_cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        expert_data_path: str,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        state_dict_path: Optional[str] = None,
) -> None:
    r"""
    Overview:
        Collect episodic demonstration data by the trained policy.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - collect_count (:obj:`int`): The count of collected data.
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
        - state_dict_path (:obj:'str') the abs path of the state dict
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(
        cfg,
        collector=EpisodeSerialCollector,
        seed=seed,
        env=env_fn,
        auto=True,
        create_cfg=create_cfg,
        save_cfg=True,
        save_path='collect_demo_data_config.py'
    )

    # Create components: env, policy, collector
    if env_setting is None:
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env, eval_=False)
    else:
        env_fn, collector_env_cfg, _ = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['collect', 'eval'])
    collect_demo_policy = policy.collect_mode
    if state_dict is None:
        assert state_dict_path is not None
        state_dict = torch.load(state_dict_path, map_location='cpu')
    policy.collect_mode.load_state_dict(state_dict)
    collector = EpisodeSerialCollector(cfg.policy.collect.collector, collector_env, collect_demo_policy)

    if hasattr(cfg.policy.other, 'eps'):
        policy_kwargs = {'eps': 0.}
    else:
        policy_kwargs = None

    # Let's collect some expert demonstrations
    exp_data = collector.collect(n_episode=collect_count, policy_kwargs=policy_kwargs)
    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    # Save data transitions.
    offline_data_save_type(exp_data, expert_data_path, data_type=cfg.policy.collect.get('data_type', 'naive'))
    print('Collect episodic demo data successfully')


def episode_to_transitions(data_path: str, expert_data_path: str, nstep: int) -> None:
    r"""
    Overview:
        Transfer episodic data into nstep transitions.
    Arguments:
        - data_path (:obj:str): data path that stores the pkl file
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - nstep (:obj:`int`): {s_{t}, a_{t}, s_{t+n}}.

    """
    with open(data_path, 'rb') as f:
        _dict = pickle.load(f)  # class is list; length is cfg.reward_model.collect_count
    post_process_data = []
    for i in range(len(_dict)):
        data = get_nstep_return_data(_dict[i], nstep)
        post_process_data.extend(data)
    offline_data_save_type(
        post_process_data,
        expert_data_path,
    )


def episode_to_transitions_filter(data_path: str, expert_data_path: str, nstep: int, min_episode_return: int) -> None:
    r"""
    Overview:
        Transfer episodic data into n-step transitions and only take the episode data whose return is larger than
        min_episode_return.
    Arguments:
        - data_path (:obj:str): data path that stores the pkl file
        - expert_data_path (:obj:`str`): File path of the expert demo data will be written to.
        - nstep (:obj:`int`): {s_{t}, a_{t}, s_{t+n}}.

    """
    with open(data_path, 'rb') as f:
        _dict = pickle.load(f)  # class is list; length is cfg.reward_model.collect_count
    post_process_data = []
    for i in range(len(_dict)):
        episode_rewards = torch.stack([_dict[i][j]['reward'] for j in range(_dict[i].__len__())], axis=0)
        if episode_rewards.sum() < min_episode_return:
            continue
        data = get_nstep_return_data(_dict[i], nstep)
        post_process_data.extend(data)
    offline_data_save_type(
        post_process_data,
        expert_data_path,
    )
