from easydict import EasyDict

cfg = dict(
    exp_name='HalfCheetah-v3-PG',
    seed=0,
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
        env_wrapper='mujoco_default',
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            entropy_weight=0.001,
        ),
        collect=dict(
            n_episode=20,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1, ))
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
