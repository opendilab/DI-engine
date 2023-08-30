from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='SpaceInvadersNoFrameskip-v4-DQN',
    seed=0,
    env=dict(
        env_id='SpaceInvadersNoFrameskip-v4',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        fram_stack=4,
        stop_value=2000,
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
            hook=dict(save_ckpt_after_iter=1000000, )
        ),
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        collect=dict(n_sample=100, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ), replay_buffer=dict(replay_buffer_size=400000, )
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
