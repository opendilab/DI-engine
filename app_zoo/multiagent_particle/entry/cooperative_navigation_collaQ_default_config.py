from easydict import EasyDict

agent_num = 5
num_agents = agent_num
num_landmarks = agent_num
actor_env_num = 4
evaluator_env_num = 2
cooperative_navigation_collaQ_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.multiagent_particle.envs.particle_env'],
        env_type='cooperative_navigation',
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        agent_num=agent_num,
        actor_env_num=actor_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='collaQ',
        import_names=['nervex.policy.collaQ'],
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=2 + 2 + (agent_num - 1) * 2 + num_landmarks * 2,
            obs_alone_dim=2 + 2 + (num_landmarks) * 2,
            global_obs_dim=agent_num * 2 + num_landmarks * 2 + agent_num * 2,
            action_dim=5,
            embedding_dim=64,
            enable_attention=True,
            self_feature_range=[2, 4],  # placeholder
            ally_feature_range=[4, 12],  # placeholder
            attention_dim=32,
        ),
        learn=dict(
            train_step=100,
            batch_size=32,
            agent_num=agent_num,
            learning_rate=0.0001,
            weight_decay=0.0001,
            algo=dict(
                target_update_theta=0.001,
                discount_factor=0.99,
            ),
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=16,
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
            decay=100000,
        ), ),
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
        collect_print_freq=4,
    ),
    evaluator=dict(
        n_episode=2,
        eval_freq=200,
        stop_val=0,  # We don't have a stop_val yet. The stop_val here is unreachable.
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
cooperative_navigation_collaQ_default_config = EasyDict(cooperative_navigation_collaQ_default_config)
main_config = cooperative_navigation_collaQ_default_config
