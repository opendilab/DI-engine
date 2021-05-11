from easydict import EasyDict

nstep = 3
cartpole_rainbowdqn_default_config = dict(
    env=dict(
        manager=dict(type='base', ),
        env_kwargs=dict(
            import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
            env_type='cartpole',
            collector_env_num=8,
            evaluator_env_num=5,
        ),
    ),
    policy=dict(
        use_cuda=False,
        policy_type='rainbow_dqn',
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=4,
            action_dim=2,
            hidden_dim_list=[128, 128, 64],
            v_max=10,
            v_min=-10,
            n_atom=51,
        ),
        learn=dict(
            train_iteration=3,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            algo=dict(
                target_update_freq=100,
                discount_factor=0.97,
                nstep=nstep,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        other=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), ),
    ),
    replay_buffer=dict(replay_buffer_size=20000, ),
    collector=dict(
        n_sample=80,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=200,
        stop_value=195,
    ),
    learner=dict(
        load_path='',
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=100, ),
            ),
        ),
    ),
)
cartpole_rainbowdqn_default_config = EasyDict(cartpole_rainbowdqn_default_config)
main_config = cartpole_rainbowdqn_default_config
