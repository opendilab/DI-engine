from easydict import EasyDict

cartpole_a2c_default_config = dict(
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
        policy_type='a2c',
        on_policy=True,
        model=dict(
            obs_dim=4,
            action_dim=2,
            embedding_dim=64,
        ),
        learn=dict(
            train_iteration=5,
            batch_size=64,
            learning_rate=0.0003,
            weight_decay=0.0,
            algo=dict(
                use_adv_norm=True,
                value_weight=0.5,
                entropy_weight=0.01,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(
                discount_factor=0.9,
                gae_lambda=0.95,
                use_nstep_return=False,
            ),
        ),
    ),
    replay_buffer=dict(replay_buffer_size=10000, ),
    collector=dict(
        n_sample=128,
        traj_len='inf',
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=50,
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
cartpole_a2c_default_config = EasyDict(cartpole_a2c_default_config)
main_config = cartpole_a2c_default_config
