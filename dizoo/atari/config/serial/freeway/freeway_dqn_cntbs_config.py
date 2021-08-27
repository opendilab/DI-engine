from copy import deepcopy
from ding import reward_model
from ding.entry import serial_pipeline, serial_pipeline_reward_model
from easydict import EasyDict

freeway_dqn_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000,
        env_id='FreewayNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.01,
                end=0.01,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=500000, ),
        ),
    ),
    reward_model=dict(
        type='countbased',
        counter_type='AutoEncoder',
        bonus_coefficent=0.1,
        state_dim=[1, 84, 84],
        hash_dim=64,
        max_buff_len=500000,
        batch_size=64,
        update_per_iter=3,
        learning_rate=0.01,
    )
)
freeway_dqn_config = EasyDict(freeway_dqn_config)
main_config = freeway_dqn_config
freeway_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
    reward_model=dict(type='countbased')
)
freeway_dqn_create_config = EasyDict(freeway_dqn_create_config)
create_config = freeway_dqn_create_config

if __name__ == '__main__':
    serial_pipeline_reward_model((main_config, create_config), seed=0)
