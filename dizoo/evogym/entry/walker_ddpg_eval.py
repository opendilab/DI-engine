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

walker_ddpg_config = dict(
    exp_name='evogym_walker_ddpg_seed0',
    env=dict(
        env_id='Walker-v0',
        robot='speed_bot',
        robot_dir='./dizoo/evogym/envs',
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=-0.5,
        manager=dict(shared_memory=False, ),
        # The path to save the game replay
        replay_path='./evogym_walker_ddpg_seed0/video',
    ),
    policy=dict(
        cuda=True,
        load_path="./evogym_walker_ddpg_seed0/ckpt/ckpt_best.pth.tar",
        random_collect_size=1000,
        model=dict(
            obs_shape=58,
            action_shape=10,
            twin_critic=False,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,  # discount_factor: 0.97-0.99
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
walker_ddpg_config = EasyDict(walker_ddpg_config)
main_config = walker_ddpg_config

walker_ddpg_create_config = dict(
    env=dict(
        type='evogym',
        import_names=['dizoo.evogym.envs.evogym_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
walker_ddpg_create_config = EasyDict(walker_ddpg_create_config)
create_config = walker_ddpg_create_config

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
