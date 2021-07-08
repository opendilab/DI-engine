from easydict import EasyDict

n_agent = 5
collector_env_num = 4
evaluator_env_num = 5
communication = True
cooperative_navigation_atoc_config = dict(
    env=dict(
        n_agent=n_agent,
        num_landmarks=n_agent,
        max_step=100,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        agent_obs_only=True,
        discrete_action=False,
        n_evaluator_episode=5,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=False,
        model=dict(
            obs_shape=2 + 2 + (n_agent - 1) * 2 + n_agent * 2,
            action_shape=5,
            n_agent=n_agent,
            communication=communication,
            thought_size=16,
            agent_per_group=min(n_agent // 2, 5),
        ),
        learn=dict(
            update_per_collect=5,
            batch_size=32,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.9,
            communication=communication,
            actor_update_freq=1,
            noise=True,
            noise_sigma=0.15,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=500,
            unroll_len=1,
            noise_sigma=0.4,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), )
    ),
)
cooperative_navigation_atoc_config = EasyDict(cooperative_navigation_atoc_config)
main_config = cooperative_navigation_atoc_config
cooperative_navigation_atoc_create_config = dict(
    env=dict(
        import_names=['dizoo.multiagent_particle.envs.particle_env'],
        type='cooperative_navigation',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='atoc'),
)
cooperative_navigation_atoc_create_config = EasyDict(cooperative_navigation_atoc_create_config)
create_config = cooperative_navigation_atoc_create_config
