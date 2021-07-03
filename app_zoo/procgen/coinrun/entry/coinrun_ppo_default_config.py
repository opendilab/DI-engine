from easydict import EasyDict

coinrun_ppo_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.procgen.coinrun.envs.coinrun_env'],
        env_type='coinrun',
        # frame_stack=4,
        is_train=True,
        collector_env_num=16,
        evaluator_env_num=8,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='ppo',
        import_names=['ding.policy.ppo'],
        on_policy=False,
        model=dict(
            model_type='conv_vac',
            import_names=['ding.model.actor_critic'],
            obs_dim=[3, 64, 64],
            action_dim=1,
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
    commander=dict(),
)
coinrun_ppo_default_config = EasyDict(coinrun_ppo_default_config)
main_config = coinrun_ppo_default_config
