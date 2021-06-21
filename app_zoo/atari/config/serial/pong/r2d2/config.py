from easydict import EasyDict
from nervex.entry import serial_pipeline
import os
import torch

collector_env_num = 16
evaluator_env_num = 2
pong_r2d2_config = dict(
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=2,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size_list=[128, 128, 512],
            head_kwargs=dict(head_type='base', ),
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0005,
            target_update_freq=400,
        ),
        collect=dict(
            n_sample=128,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
                replay_start_size=1000,
            )
        ),
    ),
)
pong_r2d2_config = EasyDict(pong_r2d2_config)
main_config = pong_r2d2_config
pong_r2d2_create_config = dict(
    env=dict(
        type='atari',
        import_names=['app_zoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2'),
)
pong_r2d2_create_config = EasyDict(pong_r2d2_create_config)
create_config = pong_r2d2_create_config

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    serial_pipeline((main_config, create_config), seed=0)