from easydict import EasyDict

agent_num = 8
collector_env_num = 8
evaluator_env_num = 5
smac_collaQ_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        env_type='smac',
        map_name='3s5z',
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        shared_memory=False,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='collaq',
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_dim=248,
            obs_alone_dim=216,
            global_obs_dim=216,
            action_dim=14,
            hidden_dim_list=[128, 128, 256],
            enable_attention=True,
            self_feature_range=[128, 136],  # placeholder
            ally_feature_range=[136, 192],  # placeholder
            attention_dim=32,
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
            traj_len='inf',
            unroll_len=16,
            agent_num=agent_num,
            env_num=collector_env_num,
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
            replay_buffer_size=5000,
            max_use=10,
        ),
    ),
    collector=dict(
        n_episode=4,
        traj_len=1000,  # smac_episode_max_length
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=50,
        stop_value=0.7,
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
smac_collaQ_default_config = EasyDict(smac_collaQ_default_config)
main_config = smac_collaQ_default_config
