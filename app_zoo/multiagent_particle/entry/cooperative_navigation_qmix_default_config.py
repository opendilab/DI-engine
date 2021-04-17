from easydict import EasyDict

agent_num = 5
collector_env_num = 4
evaluator_env_num = 2
num_agents = agent_num
num_landmarks = agent_num
max_step = 100
cooperative_navigation_qmix_default_config = dict(
    env=dict(
        manager=dict(
            type='subprocess',
            shared_memory=False,
        ),
        env_kwargs=dict(
            import_names=['app_zoo.multiagent_particle.envs.particle_env'],
            env_type='cooperative_navigation',
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            max_step=max_step,
            agent_num=agent_num,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    ),
    policy=dict(
        use_cuda=True,
        policy_type='qmix',
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=2 + 2 + (agent_num - 1) * 2 + num_landmarks * 2,
            global_obs_dim=agent_num * 2 + num_landmarks * 2 + agent_num * 2,
            action_dim=5,
            hidden_dim_list=[128, 128, 64],
        ),
        learn=dict(
            train_iteration=100,
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
            unroll_len=16,
            agent_num=agent_num,
            env_num=collector_env_num,
        ),
        eval=dict(
            agent_num=agent_num,
            env_num=evaluator_env_num,
        ),
        other=dict(eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=100000,
        ), ),
    ),
    replay_buffer=dict(
        replay_buffer_size=5000,
        max_use=10,
    ),
    collector=dict(
        n_episode=6,
        traj_len='inf',
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=5,
        stop_value=0,
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
)
cooperative_navigation_qmix_default_config = EasyDict(cooperative_navigation_qmix_default_config)
main_config = cooperative_navigation_qmix_default_config
