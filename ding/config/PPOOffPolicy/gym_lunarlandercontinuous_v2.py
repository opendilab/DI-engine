from easydict import EasyDict

cfg = dict(
    exp_name='LunarLanderContinuous-v2-PPOOffPolicy',
    seed=0,
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=4,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=240,
        act_scale=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=8,
            action_shape=2,
            action_space='continuous',
            sigma_type = 'conditioned',
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            actor_head_layer_num=2,
            critic_head_layer_num=2,
            share_encoder=False,
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=640,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            nstep=1,
            nstep_return=False,
            adv_norm=True,
            value_norm=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(render=True),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=False, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
from functools import partial
env = partial(ding.envs.gym_env.env, continuous=True)
