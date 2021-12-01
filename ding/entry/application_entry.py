from typing import Union, Optional, List, Any, Tuple
import pickle
import torch
from functools import partial
import argparse
import os

from ding.config import compile_config, read_config
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, EpisodeSerialCollector
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.torch_utils import to_device
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
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(
        cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True, save_path='eval_config.py'
    )

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
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode)

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
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
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

    policy_kwargs = None if not hasattr(cfg.policy.other.get('eps', None), 'collect') \
        else {'eps': cfg.policy.other.eps.get('collect', 0.2)}

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
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
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
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
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

    policy_kwargs = None if not hasattr(cfg.policy.other.get('eps', None), 'collect') \
        else {'eps': cfg.policy.other.eps.get('collect', 0.2)}

    # Let's collect some expert demostrations
    exp_data = collector.collect(n_episode=collect_count, policy_kwargs=policy_kwargs)
    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    # Save data transitions.
    offline_data_save_type(exp_data, expert_data_path, data_type=cfg.policy.collect.get('data_type', 'naive'))
    print('Collect episodic demo data successfully')


def collect_episodic_demo_data_for_trex(
        input_cfg: Union[str, dict],
        seed: int,
        collect_count: int,
        rank: int,
        save_cfg_path: str,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        state_dict_path: Optional[str] = None,
) -> None:
    r"""
    Overview:
        Collect episodic demonstration data by the trained policy for trex specifically.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - collect_count (:obj:`int`): The count of collected data.
        - rank (:obj:`int`) the episode ranking.
        - save_cfg_path(:obj:'str') where to save the collector config
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
    env_fn = None if env_setting is None else env_setting[0]
    cfg.env.collector_env_num=1
    if not os.path.exists(save_cfg_path):
        os.mkdir(save_cfg_path)
    cfg = compile_config(
        cfg,
        collector=EpisodeSerialCollector,
        seed=seed,
        env=env_fn,
        auto=True,
        create_cfg=create_cfg,
        save_cfg=True,
        save_path=save_cfg_path + '/collect_demo_data_config.py'
    )

    # Create components: env, policy, collector
    if env_setting is None:
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
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

    policy_kwargs = None if not hasattr(cfg.policy.other, 'eps') \
        else {'eps': cfg.policy.other.eps.get('collect', 0.2)}

    # Let's collect some sub-optimal demostrations
    exp_data = collector.collect(n_episode=collect_count, policy_kwargs=policy_kwargs)
    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    # Save data transitions.
    print('Collect {}th episodic demo data successfully'.format(rank))
    return exp_data


def epsiode_to_transitions(data_path: str, expert_data_path: str, nstep: int) -> None:
    r"""
    Overview:
        Transfer episoded data into nstep transitions
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




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./dizoo/box2d/lunarlander/config/lunarlander_trex_offppo_config.py')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args


def trex_collecting_data(args=get_args()):
    if isinstance(args.cfg, str):
        cfg, create_cfg = read_config(args.cfg)
    else:
        cfg, create_cfg = args.cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    compiled_cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg, save_cfg=False)
    offline_data_path = compiled_cfg.reward_model.offline_data_path
    expert_model_path = compiled_cfg.reward_model.expert_model_path
    checkpoint_min = compiled_cfg.reward_model.checkpoint_min
    checkpoint_max = compiled_cfg.reward_model.checkpoint_max
    checkpoint_step = compiled_cfg.reward_model.checkpoint_step
    checkpoints = []
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        checkpoints.append(str(i))
    data_for_save={}
    learning_returns=[]
    learning_rewards=[]
    episodes_data=[]
    for checkpoint in checkpoints:
        model_path = expert_model_path + \
        '/ckpt/iteration_' + checkpoint + '.pth.tar'
        seed = args.seed + (int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step)
        exp_data = collect_episodic_demo_data_for_trex(args.cfg, seed, state_dict_path=model_path, save_cfg_path=offline_data_path,
                    collect_count=1, rank=(int(checkpoint)-int(checkpoint_min))// int(checkpoint_step)+1)
        data_for_save[(int(checkpoint)-int(checkpoint_min))// int(checkpoint_step)] = exp_data[0]
        obs = list(default_collate(exp_data[0])['obs'].numpy())
        learning_rewards.append(default_collate(exp_data[0])['reward'].tolist())
        sum_reward = torch.sum(default_collate(exp_data[0])['reward']).item()
        learning_returns.append(sum_reward)
        episodes_data.append(obs)
    offline_data_save_type(data_for_save, offline_data_path + '/suboptimal_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive'))
    # if not compiled_cfg.reward_model.auto:
    offline_data_save_type(episodes_data, offline_data_path+ '/episodes_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive'))
    offline_data_save_type(learning_returns, offline_data_path + '/learning_returns.pkl', data_type=cfg.policy.collect.get('data_type', 'naive'))
    offline_data_save_type(learning_rewards, offline_data_path + '/learning_rewards.pkl', data_type=cfg.policy.collect.get('data_type', 'naive'))
    offline_data_save_type(checkpoints, offline_data_path + '/checkpoints.pkl', data_type=cfg.policy.collect.get('data_type', 'naive'))
    return checkpoints, episodes_data, learning_returns, learning_rewards

if __name__ == '__main__':
    trex_collecting_data()