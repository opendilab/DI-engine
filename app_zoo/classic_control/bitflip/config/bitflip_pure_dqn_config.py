from easydict import EasyDict

n_bits = 4
bitflip_pure_dqn_config = dict(
    env=dict(
        collector_env_num=4,
        evaluator_env_num=8,
        n_bits=n_bits,
        n_evaluator_episode=16,
        stop_value=0.9,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        model=dict(
            obs_shape=2 * n_bits,
            action_shape=n_bits,
            embedding_size=64,
            dueling=True,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=12,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(
            n_episode=8,
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                type='episode',
                replay_buffer_size=50,
            ),
        ),
    ),
)
bitflip_pure_dqn_config = EasyDict(bitflip_pure_dqn_config)
main_config = bitflip_pure_dqn_config

bitflip_pure_dqn_create_config = dict(
    env=dict(
        type='bitflip',
        import_names=['app_zoo.classic_control.bitflip.envs.bitflip_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
bitflip_pure_dqn_create_config = EasyDict(bitflip_pure_dqn_create_config)
create_config = bitflip_pure_dqn_create_config
