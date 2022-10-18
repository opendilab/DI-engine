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
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed

walker_ppo_config = dict(
    exp_name='evogym_walker_ppo_seed0',
    env=dict(
        env_id='Walker-v0',
        robot='speed_bot',
        robot_dir='./dizoo/evogym/envs',
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=100,
        manager=dict(shared_memory=False, ),
        # The path to save the game replay
        replay_path='./evogym_walker_ppo_seed0/video',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        load_path="./evogym_walker_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=58,
            action_shape=10,
            action_space='continuous',
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,
            batch_size=256,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
    )
    
)
walker_ppo_config = EasyDict(walker_ppo_config)
main_config = walker_ppo_config

walker_ppo_create_config = dict(
    env=dict(
        type='evogym',
        import_names=['dizoo.evogym.envs.evogym_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
    replay_buffer=dict(type='naive', ),
)
walker_ppo_create_config = EasyDict(walker_ppo_create_config)
create_config = walker_ppo_create_config

def main(cfg, create_cfg, seed=0):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
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
    policy = PPOPolicy(cfg.policy)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    # evaluate
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    main(main_config, create_config, seed=0)
