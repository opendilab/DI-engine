from easydict import EasyDict

agent_num = 8
actor_env_num = 16
evaluator_env_num = 10
smac_3s5z_collaQ_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.smac.envs.smac_env'],
        env_type='smac',
        map_name='3s5z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        actor_env_num=actor_env_num,
        evaluator_env_num=evaluator_env_num,
        shared_memory=False,
        obs_alone=True
    ),
    policy=dict(
        use_cuda=True,
        policy_type='collaQ',
        import_names=['ding.policy.collaQ'],
        on_policy=False,
        model=dict(
            agent_num=agent_num,
            obs_dim=150,
            obs_alone_dim=94,
            global_obs_dim=216,
            action_dim=14,
            embedding_dim=64,
            enable_attention=True,
            self_feature_range=[124, 128],  # placeholder 4
            ally_feature_range=[68, 124],  # placeholder  8*7
            attention_dim=32,
        ),
        learn=dict(
            train_step=20,
            batch_size=32,
            clip=False,
            agent_num=agent_num,
            learning_rate=0.0005,
            algo=dict(
                target_update_theta=0.001,
                discount_factor=0.99,
            ),
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=20,
            agent_num=agent_num,
            env_num=actor_env_num,
        ),
        eval=dict(
            agent_num=agent_num,
            env_num=evaluator_env_num,
        ),
        command=dict(eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=200000,
        ), ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            maxlen=5000,
            max_reuse=10,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_episode=32,
        traj_len='inf',  # smac_episode_max_length
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=30,
        evaluator_env_num=evaluator_env_num,
        eval_freq=200,
        stop_val=0.999,
    ),
    learner=dict(
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=500, ),
            ),
        ),
    ),
    commander=dict(),
)
smac_3s5z_collaQ_config = EasyDict(smac_3s5z_collaQ_config)
main_config = smac_3s5z_collaQ_config
