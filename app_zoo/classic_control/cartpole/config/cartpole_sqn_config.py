from easydict import EasyDict

update_per_collect = 16
cartpole_sqn_config = dict(
    env=dict(
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        collector_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        cuda=False,
        multi_gpu=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            hidden_size_list=[128, 128, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=64,
            learning_rate_q=0.001,
            learning_rate_alpha=0.001,
            alpha=0.2,
            target_entropy=0.2,
        ),
        collect=dict(
            n_sample=update_per_collect,
            nstep=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.8,
                decay=2000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    )
)
cartpole_sqn_config = EasyDict(cartpole_sqn_config)
main_config = cartpole_sqn_config

cartpole_sqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sqn'),
)
cartpole_sqn_create_config = EasyDict(cartpole_sqn_create_config)
create_config = cartpole_sqn_create_config
