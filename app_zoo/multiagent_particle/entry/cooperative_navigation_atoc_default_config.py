from easydict import EasyDict

agent_num = 5
num_agents = agent_num
num_landmarks = agent_num
actor_env_num = 4
evaluator_env_num = 2
use_communication = False
thought_dim = 16
batch_size = 32
cooperative_navigation_atoc_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.multiagent_particle.envs.particle_env'],
        env_type='cooperative_navigation',
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        agent_num=agent_num,
        max_step=1000,
        actor_env_num=actor_env_num,
        evaluator_env_num=evaluator_env_num,
        agent_obs_only=True,
        use_discrete=False,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='atoc',
        import_names=['nervex.policy.atoc'],
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=2 + 2 + (agent_num - 1) * 2 + num_landmarks * 2,
            action_dim=5,
            thought_dim=thought_dim,
            n_agent=agent_num,
            use_communication=use_communication,
            m_group=min(agent_num // 2, 5),
            T_initiate=5,
        ),
        learn=dict(
            train_step=2,
            batch_size=batch_size,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            weight_decay=0.0001,
            ignore_done=True,
            algo=dict(
                target_theta=0.005,
                discount_factor=0.9,
                use_communication=use_communication,
                actor_update_freq=1,
                use_noise=True,
                noise_sigma=0.2,
                noise_range=dict(
                    min=-0.5,
                    max=0.5,
                ),
            ),
            init_data_count=16,
        ),
        collect=dict(
            traj_len=100,
            unroll_len=1,
            algo=dict(noise_sigma=0.1, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
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
        eval_freq=1000,
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
cooperative_navigation_atoc_default_config = EasyDict(cooperative_navigation_atoc_default_config)
main_config = cooperative_navigation_atoc_default_config
