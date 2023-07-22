from easydict import EasyDict

cfg = dict(
    exp_name='Bipedalwalker-v3-A2C',
    seed=0,
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=24,
            action_shape=4,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.0003,
            value_weight=0.7,
            entropy_weight=0.0005,
            discount_factor=0.99,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=64,
            discount_factor=0.99,
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
