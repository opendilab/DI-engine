import argparse
import copy
import pickle
import random
import easydict
import torch
import os
from typing import Optional, List, Any
from functools import partial
from copy import deepcopy

from ding.config import compile_config, read_config
from ding.worker import EpisodeSerialCollector, create_buffer, BaseLearner
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, bc
from ding.torch_utils import to_device
from ding.utils import set_pkg_seed
from ding.utils.data import default_collate, offline_data_save_type
from functools import reduce


def collect_episodic_demo_data_for_drex(
    cfg: easydict,
    seed: int,
    collect_count: int,
    rank: int,
    save_cfg_path: str,
    noise: float,
    env_setting: Optional[List[Any]] = None,
    model: Optional[torch.nn.Module] = None,
    state_dict: Optional[dict] = None,
    state_dict_path: Optional[str] = None,
):
    r"""
    Overview:
        Collect episodic demonstration data by the trained policy for trex specifically.
    Arguments:
        - cfg (:obj:`easydict`): Config in dict type.
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
    cfg.env.collector_env_num = 1
    if not os.path.exists(save_cfg_path):
        os.mkdir(save_cfg_path)

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

    policy_kwargs = {'eps': noise}

    # Let's collect some sub-optimal demonstrations
    exp_data = collector.collect(n_episode=collect_count, policy_kwargs=policy_kwargs)

    if cfg.policy.cuda:
        exp_data = to_device(exp_data, 'cpu')
    # Save data transitions.
    print('Collect {}th episodic demo data successfully'.format(rank))
    return exp_data


def drex_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='abs path for a config')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def eval_bc(validation_set, policy, use_cuda):
    tot = 0
    tot_acc = 0
    device = 'cuda' if use_cuda else 'cpu'
    for _, data in enumerate(validation_set):
        x, y = {'obs': data['obs'].to(device).squeeze(0)}, data['action'].squeeze(-1)
        y_pred = policy.forward(x, eps=-1)['obs']['action']
        tot += y_pred.shape[0]
        tot_acc += (y_pred == y).sum().item()
    acc = tot_acc / tot
    return acc


def train_bc(cfg, pre_expert_data=None, max_iterations=6000):
    cfg_new = copy.deepcopy(cfg).policy
    cfg_new.continuous = False
    bc_policy = bc.BehaviourCloningPolicy(cfg_new)

    if pre_expert_data is None:
        with open(cfg.reward_model.offline_data_path + '/suboptimal_data.pkl', 'rb') as f:
            pre_expert_data = pickle.load(f)

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer)
    push_data = []
    for i in range(len(pre_expert_data)):
        push_data += pre_expert_data[i]

    random.shuffle(push_data)
    validation_set = push_data[-len(push_data) // 10:]
    push_data = push_data[:-len(push_data) // 10]
    replay_buffer.push(push_data, cur_collector_envstep=0)
    learner = BaseLearner(cfg.policy.learn.learner, bc_policy.learn_mode)

    best_acc = 0
    cnt = 0
    for i in range(max_iterations):
        train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        if i % 100 == 0:
            acc = eval_bc(validation_set, bc_policy.collect_mode, cfg.policy.cuda)
            if acc < best_acc:
                cnt += 1
                if cnt > 100:
                    break
            else:
                cnt = 0
                best_acc = acc
                torch.save(bc_policy.collect_mode.state_dict(), cfg.reward_model.offline_data_path + '/bc_best.pth.tar')
        if train_data is None:
            replay_buffer.push(push_data, cur_collector_envstep=0)
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        learner.train(train_data)

    ckpt = torch.load(cfg.reward_model.offline_data_path + '/bc_best.pth.tar')
    bc_policy.collect_mode.load_state_dict(ckpt)
    return bc_policy


def load_bc(load_path, cfg):
    cfg_new = copy.deepcopy(cfg).policy
    cfg_new.continuous = False
    bc_policy = bc.BehaviourCloningPolicy(cfg_new)
    state_dict = torch.load(load_path, map_location='cpu')
    bc_policy.collect_mode.load_state_dict(state_dict)
    print('Load bc from {}'.format(load_path))
    return bc_policy


def cal_mean(lis):
    return reduce(lambda x, y: x + y, lis) / len(lis)


def create_data_drex(bc_policy, cfg):
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(copy.deepcopy(cfg.env))
    collector_env = create_env_manager(
        copy.deepcopy(cfg.env.manager), [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    collector = EpisodeSerialCollector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=bc_policy.collect_mode,
    )
    eps_list = cfg.reward_model.eps_list
    eps_list.sort(reverse=True)
    created_data = []
    created_data_returns = []
    for eps in eps_list:
        policy_kwargs = {'eps': eps}
        # Let's collect some sub-optimal demonstrations
        exp_data = collector.collect(n_episode=cfg.reward_model.num_trajs_per_bin, policy_kwargs=policy_kwargs)
        episodes = [default_collate(data)['obs'].numpy() for data in exp_data]
        returns = [torch.sum(default_collate(data)['reward']).item() for data in exp_data]

        created_data.append(episodes)
        created_data_returns.append(returns)
        print('noise: {}, returns: {}, avg: {}'.format(eps, returns, cal_mean(returns)))
    return created_data, created_data_returns


def drex_generating_data(compiled_cfg):
    offline_data_path = compiled_cfg.reward_model.offline_data_path
    expert_model_path = compiled_cfg.reward_model.expert_model_path
    data_for_save = {}
    learning_returns = []
    learning_rewards = []
    episodes_data = []
    for i in range(10):
        model_path = expert_model_path
        seed = compiled_cfg.seed + i
        exp_data = collect_episodic_demo_data_for_drex(
            deepcopy(compiled_cfg),
            seed,
            noise=-1,
            state_dict_path=model_path,
            save_cfg_path=offline_data_path,
            collect_count=1,
            rank=i
        )
        data_for_save[i] = exp_data[0]
        obs = list(default_collate(exp_data[0])['obs'].numpy())
        learning_rewards.append(default_collate(exp_data[0])['reward'].tolist())
        sum_reward = torch.sum(default_collate(exp_data[0])['reward']).item()
        learning_returns.append(sum_reward)
        episodes_data.append(obs)

    offline_data_save_type(
        data_for_save,
        offline_data_path + '/suboptimal_data.pkl',
        data_type=compiled_cfg.policy.collect.get('data_type', 'naive')
    )

    return data_for_save


def drex_collecting_data(args=drex_get_args(), seed=0):
    if isinstance(args.cfg, str):
        cfg, create_cfg = read_config(args.cfg)
    else:
        cfg, create_cfg = deepcopy(args.cfg)
    bc_iteration = getattr(cfg, 'bc_iteration', 50000)
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    pre_expert_data = drex_generating_data(cfg)
    returns = [torch.sum(default_collate(pre_expert_data[i])['reward']).item() for i in range(len(pre_expert_data))]
    print('demonstrations rewards: ' + str(returns))

    if 'bc_path' in cfg.reward_model and os.path.exists(cfg.reward_model.bc_path):
        bc_policy = load_bc(cfg.reward_model.bc_path, cfg)
    else:
        bc_policy = train_bc(cfg=cfg, pre_expert_data=pre_expert_data, max_iterations=bc_iteration)
    created_data, created_data_returns = create_data_drex(bc_policy, cfg)
    offline_data_path = cfg.reward_model.offline_data_path

    offline_data_save_type(
        created_data, offline_data_path + '/episodes_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )

    offline_data_save_type(
        created_data_returns,
        offline_data_path + '/learning_returns.pkl',
        data_type=cfg.policy.collect.get('data_type', 'naive')
    )


if __name__ == '__main__':
    drex_collecting_data()
