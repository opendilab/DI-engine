from easydict import EasyDict

cfg = dict(
    exp_name='PongNoFrameskip-v4-DQN',
    seed=0,
    env=dict(
        env_id='PongNoFrameskip-v4',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        fram_stack=4,
        env_wrapper='atari_default',
    ),
    policy=dict(
        cuda=True,
        priority=False,
        discount_factor=0.99,
        nstep=3,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            # Frequency of target network update.
            target_update_freq=500,
        ),
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        collect=dict(n_sample=96, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
