from easydict import EasyDict

sumo_ppo_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.sumo.envs.sumo_env'],
        env_type='sumo_wj3',
        collector_env_num=4,
        evaluator_env_num=1,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='md_ppo',
        import_names=['app_zoo.common.policy.md_ppo'],
        on_policy=False,
        model=dict(
            obs_dim=380,
            action_dim=[2, 2, 3],
            embedding_dim=64,
        ),
        learn=dict(
            train_iteration=5,
            batch_size=64,
            learning_rate=0.001,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.2,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(
                discount_factor=0.9,
                gae_lambda=0.95,
            ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=1000,
            max_use=100,
        ),
    ),
    collector=dict(
        n_sample=16,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=1,
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
    commander=dict(),
)
sumo_ppo_default_config = EasyDict(sumo_ppo_default_config)
main_config = sumo_ppo_default_config
