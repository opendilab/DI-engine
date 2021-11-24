from easydict import EasyDict

cartpole_rainbow_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        priority=True,
        discount_factor=0.97,
        nstep=3,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=80,
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
)
cartpole_rainbow_config = EasyDict(cartpole_rainbow_config)
main_config = cartpole_rainbow_config
cartpole_rainbow_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='rainbow'),
)
cartpole_rainbow_create_config = EasyDict(cartpole_rainbow_create_config)
create_config = cartpole_rainbow_create_config
