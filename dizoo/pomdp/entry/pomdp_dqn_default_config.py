from ding.entry import serial_pipeline
from easydict import EasyDict

pong_dqn_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-ramNoFrameskip-v4',
        frame_stack=4,
        warp_frame=False,
        use_ram=True,
        pomdp=dict(noise_scale=0.01, zero_p=0.2, reward_noise=0.01, duplicate_p=0.2),
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[
                512,
            ],
            action_shape=6,
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
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_dqn_config = EasyDict(pong_dqn_config)
main_config = pong_dqn_config
pong_dqn_create_config = dict(
    env=dict(
        type='pomdp',
        import_names=['app_zoo.pomdp.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
pong_dqn_create_config = EasyDict(pong_dqn_create_config)
create_config = pong_dqn_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
