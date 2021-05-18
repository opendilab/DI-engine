from easydict import EasyDict

bitflip_dqn_config = dict(
    env=dict(
        collector_env_num=1,
        evaluator_env_num=8,
        n_bits=5,
        evaluator_n_episode=16,
        stop_value=0.9,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        model=dict(
            obs_shape=10,
            action_shape=5,
            embedding_size=64,
            dueling=True,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.0001,
            nstep=1,
            target_update_freq=500,
            discount_factor=0.9,
        ),
        collect=dict(
            n_episode=1,
            n_sample=None,
            unroll_len=1,
            nstep=1,
            her=True,
            her_strategy='final',
            her_replay_k=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=5000, ),
        ),
    ),
)
bitflip_dqn_config = EasyDict(bitflip_dqn_config)
main_config = bitflip_dqn_config
bitflip_dqn_create_config = dict(
    env=dict(
        type='bitflip',
        import_names=['app_zoo.classic_control.bitflip.envs.bitflip_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
bitflip_dqn_create_config = EasyDict(bitflip_dqn_create_config)
create_config = bitflip_dqn_create_config
