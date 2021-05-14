from easydict import EasyDict

discount_factor = 0.99
qbert_ppo_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        collector_env_num=8,
        evaluator_env_num=3,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='ppo',
        import_names=['nervex.policy.ppo'],
        on_policy=True,  # Simulate parallel
        model=dict(
            model_type='conv_vac',
            import_names=['nervex.model.actor_critic'],
            obs_dim=[4, 84, 84],
            action_dim=6,
            embedding_dim=128,
        ),
        learn=dict(
            train_iteration=16,
            batch_size=128,
            learning_rate=0.0001,
            weight_decay=0.0,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.1,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(discount_factor=discount_factor, gae_lambda=0.95),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
            max_use=3,
        ),
    ),
    collector=dict(
        n_sample=1024,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=3,
        eval_freq=1000,
        stop_value=40000,
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
qbert_ppo_default_config = EasyDict(qbert_ppo_default_config)
main_config = qbert_ppo_default_config
