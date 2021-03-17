from easydict import EasyDict

traj_len = 6
qbert_a2c_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        actor_env_num=16,
        evaluator_env_num=8,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='a2c',
        import_names=['nervex.policy.a2c'],
        on_policy=True,
        model=dict(
            model_type='conv_vac',
            import_names=['nervex.model.actor_critic'],
            obs_dim=[4, 84, 84],
            action_dim=6,
            embedding_dim=128,
        ),
        learn=dict(
            train_step=1,
            batch_size=80,
            learning_rate=0.0001,
            weight_decay=0.0,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
            ),
        ),
        collect=dict(
            traj_len=traj_len,
            unroll_len=1,
            algo=dict(
                gae_lambda=0.99,
                discount_factor=0.99,
            ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=10000,
            max_reuse=1,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=80,
        traj_len=traj_len,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=10000,
        stop_val=9000,
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
qbert_a2c_default_config = EasyDict(qbert_a2c_default_config)
main_config = qbert_a2c_default_config
