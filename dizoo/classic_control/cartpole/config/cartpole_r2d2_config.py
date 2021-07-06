from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
cartpole_r2d2_config = dict(
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        discount_factor=0.997,
        burnin_step=5,
        nstep=5,
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.0005,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=100000,
            ), replay_buffer=dict(replay_buffer_size=5000, )
        ),
    ),
)
cartpole_r2d2_config = EasyDict(cartpole_r2d2_config)
main_config = cartpole_r2d2_config
cartpole_r2d2_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
cartpole_r2d2_create_config = EasyDict(cartpole_r2d2_create_config)
create_config = cartpole_r2d2_create_config
