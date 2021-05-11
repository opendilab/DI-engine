from easydict import EasyDict

discount_factor = 0.9
pong_impala_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        collector_env_num=16,
        evaluator_env_num=8,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='impala',
        import_names=['nervex.policy.impala'],
        on_policy=False,
        model=dict(
            model_type='conv_vac',
            import_names=['nervex.model.actor_critic'],
            obs_dim=[4, 84, 84],
            action_dim=6,
            embedding_dim=512,
        ),
        learn=dict(
            train_iteration=4,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.0001,
            grad_clip_type='clip_norm',
            clip_value=0.5,
            unroll_len=64,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
                discount_factor=discount_factor,
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
            algo=dict(discount_factor=discount_factor, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=10000,
            max_use=100,
        ),
    ),
    collector=dict(
        n_sample=8,
        traj_len=64,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=100,
        stop_value=20,
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
pong_impala_default_config = EasyDict(pong_impala_default_config)
main_config = pong_impala_default_config
