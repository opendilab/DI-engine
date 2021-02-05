from easydict import EasyDict

agent_num = 8
smac_coma_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.smac.envs.smac_env'],
        env_type='smac',
        map_name='3s5z',
        agent_num=agent_num,
        actor_env_num=8,
        evaluator_env_num=5,
        shared_memory=False,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='coma',
        import_names=['nervex.policy.coma'],
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=dict(
                agent_state=[8, 248],
                global_state=216,
            ),
            act_dim=[14, ],
            embedding_dim=64,
        ),
        learn=dict(
            train_step=1,
            batch_size=32,
            agent_num=agent_num，
            learning_rate=0.0005,
            weight_decay=0.00001,
            algo=dict(
                target_theta=0.001,
                discount_factor=0.99,
                td_lambda=0.8,
                value_weight=1.0,
                entropy_weight=0.01,
            ),
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=16,
            agent_num=agent_num,
        ),
        eval=dict(
            agent_num=agent_num,
        ),
        command=dict(
            eps=dict(
                type='exp',
                start=0.5,
                end=0.01,
                decay=100000,
            ),
        ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=64,
            max_reuse=100,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_episode=4,
        traj_len=1000,  # smac_episode_max_length
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=3,
        eval_freq=1000,
        stop_val= 0.7,
    ),
    learner=dict(
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(
                    freq=500,
                ),
            ),
        ),
    ),
    commander=dict(),
)
smac_coma_default_config = EasyDict(smac_coma_default_config)
