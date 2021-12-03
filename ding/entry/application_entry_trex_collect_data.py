import argparse
import torch

from ding.config import compile_config, read_config
from ding.utils.data import offline_data_save_type
from ding.utils.data import default_collate
from ding.entry import collect_episodic_demo_data_for_trex


def trex_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='abs path for a config')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def trex_collecting_data(args=trex_get_args()):
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
    data_for_save = {}
    learning_returns = []
    learning_rewards = []
    episodes_data = []
    for checkpoint in checkpoints:
        model_path = expert_model_path + \
        '/ckpt/iteration_' + checkpoint + '.pth.tar'
        seed = args.seed + (int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step)
        exp_data = collect_episodic_demo_data_for_trex(
            args.cfg,
            seed,
            state_dict_path=model_path,
            save_cfg_path=offline_data_path,
            collect_count=1,
            rank=(int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step) + 1
        )
        data_for_save[(int(checkpoint) - int(checkpoint_min)) // int(checkpoint_step)] = exp_data[0]
        obs = list(default_collate(exp_data[0])['obs'].numpy())
        learning_rewards.append(default_collate(exp_data[0])['reward'].tolist())
        sum_reward = torch.sum(default_collate(exp_data[0])['reward']).item()
        learning_returns.append(sum_reward)
        episodes_data.append(obs)
    offline_data_save_type(
        data_for_save,
        offline_data_path + '/suboptimal_data.pkl',
        data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    # if not compiled_cfg.reward_model.auto: more feature
    offline_data_save_type(
        episodes_data, offline_data_path + '/episodes_data.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        learning_returns,
        offline_data_path + '/learning_returns.pkl',
        data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        learning_rewards,
        offline_data_path + '/learning_rewards.pkl',
        data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    offline_data_save_type(
        checkpoints, offline_data_path + '/checkpoints.pkl', data_type=cfg.policy.collect.get('data_type', 'naive')
    )
    return checkpoints, episodes_data, learning_returns, learning_rewards


if __name__ == '__main__':
    trex_collecting_data()
