from copy import deepcopy
from nervex.entry import serial_pipeline
from easydict import EasyDict

space_invaders_rainbow_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='SpaceInvadersNoFrameskip-v4',
        frame_stack=4,
        manager=dict(
            shared_memory=False,
        )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size_list=[128, 128, 512],
            head_kwargs=dict(head_type='rainbow', ),
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            iqn=False,
        ),
        collect=dict(
            n_sample=100,
        ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.05,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
space_invaders_rainbow_config = EasyDict(space_invaders_rainbow_config)
main_config = space_invaders_rainbow_config
space_invaders_rainbow_create_config = dict(
    env=dict(
        type='atari',
        import_names=['app_zoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
space_invaders_rainbow_create_config = EasyDict(space_invaders_rainbow_create_config)
create_config = space_invaders_rainbow_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
