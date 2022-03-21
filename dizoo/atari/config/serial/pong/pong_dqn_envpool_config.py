from easydict import EasyDict

pong_dqn_envpool_config = dict(
    exp_name='pong_dqn_envpool',
    env=dict(
        collector_env_num=8,
        collector_batch_size=8,
        evaluator_env_num=8,
        evaluator_batch_size=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-v5',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
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
        collect=dict(n_sample=96, ),
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
pong_dqn_envpool_config = EasyDict(pong_dqn_envpool_config)
main_config = pong_dqn_envpool_config
pong_dqn_envpool_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='env_pool'),
    policy=dict(type='dqn'),
    # replay_buffer=dict(type='deque'),
)
pong_dqn_envpool_create_config = EasyDict(pong_dqn_envpool_create_config)
create_config = pong_dqn_envpool_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)

# Alternatively, one can be opt to run the following command to directly execute this config file
# ding -m serial -c pong_dqn_envpool_config.py -s 0