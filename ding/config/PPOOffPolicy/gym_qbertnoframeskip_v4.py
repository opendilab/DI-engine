from easydict import EasyDict

cfg = dict(
    exp_name='QbertNoFrameskip-v4-PPOOffPolicy',
    env=dict(
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        env_wrapper='atari_default',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[32, 64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            critic_head_layer_num=2,
        ),
        learn=dict(
            update_per_collect=18,
            batch_size=128,
            learning_rate=0.0001,
            value_weight=1.0,
            entropy_weight=0.005,
            clip_ratio=0.1,
            adv_norm=False,
        ),
        collect=dict(
            n_sample=1024,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
