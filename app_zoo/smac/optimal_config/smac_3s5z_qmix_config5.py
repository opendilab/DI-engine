from easydict import EasyDict

agent_num = 8
actor_env_num = 16
evaluator_env_num = 10
smac_3s5z_qmix_config5 = dict(
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
    ),
    policy=dict(
        use_cuda=True,
        policy_type='qmix',
        import_names=['nervex.policy.qmix'],
        on_policy=False,
        model=dict(
            agent_num=agent_num,
            obs_dim=150,
            global_obs_dim=216,
            action_dim=14,
            embedding_dim=64,
        ),
        learn=dict(
            train_step=20,
            batch_size=32,
            agent_num=agent_num,
            learning_rate=0.0005,
            weight_decay=0.0001,
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
            start=1,
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
smac_3s5z_qmix_config5 = EasyDict(smac_3s5z_qmix_config5)
main_config = smac_3s5z_qmix_config5
