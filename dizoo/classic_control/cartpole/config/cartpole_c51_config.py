from easydict import EasyDict

cartpole_c51_config = dict(
    exp_name='cartpole_c51',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        priority=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        discount_factor=0.97,
        nstep=3,
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
cartpole_c51_config = EasyDict(cartpole_c51_config)
main_config = cartpole_c51_config
cartpole_c51_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='c51'),
)
cartpole_c51_create_config = EasyDict(cartpole_c51_create_config)
create_config = cartpole_c51_create_config
