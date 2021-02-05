from easydict import EasyDict

agent_num = 5
cooperative_navigation_qmix_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.multiagent_particle.envs.particle_env'],
        env_type='cooperative_navigation',
        num_agents=5,
        num_landmarks=5,
        agent_num=agent_num,
        actor_env_num=8,
        evaluator_env_num=3,
        shared_memory=False,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='qmix',
        import_names=['nervex.policy.qmix'],
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=22,
            global_obs_dim=30,
            action_dim=5,
            embedding_dim=64,
        ),
        learn=dict(
            train_step=100,
            batch_size=32,
            agent_num=agent_num，
            learning_rate=0.0005,
            weight_decay=0.0001,
            algo=dict(
                target_theta=0.001,
                discount_factor=0.99,
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
                start=1.0,
                end=0.05,
                decay=100000,
            ),
        ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=5000,
            max_reuse=10,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_episode=4,
        traj_len=100,  # cooperative_navigation_episode_max_length
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=5,
        stop_val=0,
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
cooperative_navigation_qmix_default_config = EasyDict(cooperative_navigation_qmix_default_config)
