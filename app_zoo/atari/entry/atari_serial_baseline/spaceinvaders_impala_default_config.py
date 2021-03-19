from easydict import EasyDict

discount_factor = 0.99
spaceinvaders_impala_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='SpaceInvadersNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        actor_env_num=4,
        evaluator_env_num=4,
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
            train_step=4,
            batch_size=32,
            learning_rate=0.0003,
            weight_decay=0.0,
            optim='adam',
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
            meta_maxlen=10000,
            max_reuse=100,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=16,
        traj_len=128,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=100,
        stop_val=10000,
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
spaceinvaders_impala_default_config = EasyDict(spaceinvaders_impala_default_config)
main_config = spaceinvaders_impala_default_config
