from easydict import EasyDict

update_per_collect = 8
cartpole_sqn_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=update_per_collect,
            batch_size=64,
            learning_rate_q=0.001,
            learning_rate_alpha=0.001,
            alpha=0.2,
            target_entropy=0.2,
        ),
        collect=dict(
            n_sample=update_per_collect * 2,
            nstep=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.8,
                decay=2000,
            ), replay_buffer=dict(replay_buffer_size=10000, )
        ),
    )
)
cartpole_sqn_config = EasyDict(cartpole_sqn_config)
main_config = cartpole_sqn_config

cartpole_sqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sqn'),
)
cartpole_sqn_create_config = EasyDict(cartpole_sqn_create_config)
create_config = cartpole_sqn_create_config
