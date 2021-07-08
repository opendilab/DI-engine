from typing import Union, Optional, List, Any, Callable, Tuple
import pickle
import torch
from functools import partial

from ding.config import compile_config, read_config
from ding.worker import BaseLearner, SampleCollector, BaseSerialEvaluator
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.torch_utils import to_device
from ding.utils import set_pkg_seed


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
        cfg, create_cfg = input_cfg
    # TODO when env_setting is not None
    assert env_setting is None  # temporally
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)

    # Create components: env, policy, evaluator
    if env_setting is None:
        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
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
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode)

    # Evaluate
    _, eval_reward = evaluator.eval()
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))
    return eval_reward


def collect_demo_data(
        input_cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        expert_data_path: str,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
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
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    # TODO when env_setting is not None
    assert env_setting is None  # temporally
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)

    # Create components: env, policy, collector
    if env_setting is None:
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, _ = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['collect', 'eval'])
    # for policies like DQN (in collect_mode has eps-greedy)
    # collect_demo_policy = policy.collect_function(
    #     policy._data_preprocess_collect,
    #     # forward_collect -> forward_eval, because eps-greedy exploration is not needed.
    #     policy._forward_eval,
    #     policy._data_postprocess_collect,
    #     policy._process_transition,
    #     policy._get_train_sample,
    #     policy._reset_collect,
    #     policy.get_attribute,
    #     policy.set_attribute,
    #     policy.state_dict_handle,
    # )
    collect_demo_policy = policy.collect_mode
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.collect_mode.load_state_dict(state_dict)
    collector = SampleCollector(cfg.policy.collect.collector, collector_env, collect_demo_policy)

    # Let's collect some expert demostrations
    exp_data = collector.collect(n_sample=collect_count)
    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)
    print('Collect demo data successfully')
