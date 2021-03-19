from easydict import EasyDict

agent_num = 5
actor_env_num = 4
evaluator_env_num = 2
num_agents = agent_num
num_landmarks = agent_num
max_step = 100
cooperative_navigation_coma_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.multiagent_particle.envs.particle_env'],
        env_type='cooperative_navigation',
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        max_step=max_step,
        agent_num=agent_num,
        actor_env_num=actor_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='coma',
        import_names=['nervex.policy.coma'],
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=dict(
                agent_state=[agent_num, 2 + 2 + (agent_num - 1) * 2 + num_landmarks * 2],
                global_state=agent_num * 2 + num_landmarks * 2 + agent_num * 2,
            ),
            act_dim=[
                5,
            ],
            embedding_dim=64,
        ),
        learn=dict(
            train_step=1,
            batch_size=32,
            agent_num=agent_num,
            learning_rate=0.0005,
            weight_decay=0.00001,
            algo=dict(
                target_update_theta=0.001,
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
            env_num=actor_env_num,
        ),
        eval=dict(
            agent_num=agent_num,
            env_num=evaluator_env_num,
        ),
        command=dict(eps=dict(
            type='exp',
            start=0.5,
            end=0.01,
            decay=100000,
        ), ),
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
        traj_len=max_step,  # cooperative_navigation_episode_max_length
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=3,
        eval_freq=1000,
        stop_val=0,
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
cooperative_navigation_coma_default_config = EasyDict(cooperative_navigation_coma_default_config)
main_config = cooperative_navigation_coma_default_config
