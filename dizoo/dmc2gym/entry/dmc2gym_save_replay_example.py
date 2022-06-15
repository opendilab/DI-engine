import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import DDPGPolicy
from ding.utils import set_pkg_seed

cartpole_balance_ddpg_config = dict(
    exp_name='dmc2gym_cartpole_balance_ddpg_eval',
    env=dict(
        env_id='dmc2gym_cartpole_balance',
        domain_name='cartpole',
        task_name='balance',
        from_pixels=False,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        replay_path='./dmc2gym_cartpole_balance_ddpg_eval/video',
        stop_value=1000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=2560,
        load_path="./dmc2gym_cartpole_balance_ddpg/ckpt/iteration_10000.pth.tar",
        model=dict(
            obs_shape=5,
            action_shape=1,
            twin_critic=False,
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    )
)
cartpole_balance_ddpg_config = EasyDict(cartpole_balance_ddpg_config)
main_config = cartpole_balance_ddpg_config

cartpole_balance_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
cartpole_balance_create_config = EasyDict(cartpole_balance_create_config)
create_config = cartpole_balance_create_config


def main(cfg, create_cfg, seed=0):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DDPGPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        create_cfg=create_cfg,
        save_cfg=True
    )

    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.enable_save_replay(cfg.env.replay_path)

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    policy = DDPGPolicy(cfg.policy)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    # evaluate
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    main(main_config, create_config, seed=0)
