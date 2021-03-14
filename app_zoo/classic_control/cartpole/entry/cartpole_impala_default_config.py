from easydict import EasyDict

cartpole_impala_default_config = dict(
    env=dict(
        env_manager_type='base',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        actor_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='impala',
        on_policy=False,
        model=dict(
            obs_dim=4,
            action_dim=2,
            embedding_dim=64,
        ),
        learn=dict(
            train_step=5,
            batch_size=32,
            learning_rate=0.001,
            weight_decay=0.0001,
            init_data_count=600,
            unroll_len=64,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
                discount_factor=0.9,
                lambda_=0.95,
                rho_clip_ratio=1.0,
                c_clip_ratio=1.0,
                rho_pg_clip_ratio=1.0,
            ),
            ignore_done=True,
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=64,
            algo=dict(discount_factor=0.9, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=1000,
            max_reuse=100,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=16,
        traj_len=200,  # cartpole max episode len
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=200,
        stop_val=195,
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
    commander=dict(),
)
cartpole_impala_default_config = EasyDict(cartpole_impala_default_config)
main_config = cartpole_impala_default_config
