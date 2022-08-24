import argparse
import torch
import os
from typing import Union, Optional, List, Any
from functools import partial
from copy import deepcopy

from ding.config import compile_config, read_config
from ding.worker import EpisodeSerialCollector
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.torch_utils import to_device
from ding.utils import set_pkg_seed
from ding.utils.data import offline_data_save_type
from ding.utils.data import default_collate


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
):
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
        - state_dict_path (:obj:'str') the abs path of the state dict
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type += '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg.env.collector_env_num = 1
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


def trex_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='abs path for a config')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def trex_collecting_data(args=None):
    if args is None:
        args = trex_get_args()  # TODO(nyz) use sub-command in cli
    if isinstance(args.cfg, str):
        cfg, create_cfg = read_config(args.cfg)
    else:
        cfg, create_cfg = deepcopy(args.cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    compiled_cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg, save_cfg=False)
    data_path = compiled_cfg.reward_model.data_path
    expert_model_path = compiled_cfg.reward_model.expert_model_path
    checkpoint_min = compiled_cfg.reward_model.checkpoint_min
    checkpoint_max = compiled_cfg.reward_model.checkpoint_max
    checkpoint_step = compiled_cfg.reward_model.checkpoint_step
    checkpoints = []
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        checkpoints.append(str(i))
    data_for_save = {}
    learning_returns = []
    learning_rewards = []
    episodes_data = []
    for checkpoint in checkpoints:
        num_per_ckpt = 1
        model_path = expert_model_path + \
        '/ckpt/iteration_' + checkpoint + '.pth.tar'
        seed = args.seed + (int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step)
        exp_data = collect_episodic_demo_data_for_trex(
            deepcopy(args.cfg),
            seed,
            state_dict_path=model_path,
            save_cfg_path=data_path,
            collect_count=num_per_ckpt,
            rank=(int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step) + 1
        )
        data_for_save[(int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step)] = exp_data
        obs = [list(default_collate(exp_data[i])['obs'].numpy()) for i in range(len(exp_data))]
        rewards = [default_collate(exp_data[i])['reward'].tolist() for i in range(len(exp_data))]
        sum_rewards = [torch.sum(default_collate(exp_data[i])['reward']).item() for i in range(len(exp_data))]

        learning_rewards.append(rewards)
        learning_returns.append(sum_rewards)
        episodes_data.append(obs)
    offline_data_save_type(
        data_for_save, data_path + '/suboptimal_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    # if not compiled_cfg.reward_model.auto: more feature
    offline_data_save_type(
        episodes_data, data_path + '/episodes_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        learning_returns, data_path + '/learning_returns.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        learning_rewards, data_path + '/learning_rewards.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        checkpoints, data_path + '/checkpoints.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    return checkpoints, episodes_data, learning_returns, learning_rewards


if __name__ == '__main__':
    trex_collecting_data()
