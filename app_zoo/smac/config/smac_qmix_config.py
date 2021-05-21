from easydict import EasyDict

agent_num = 8

smac_qmix_config = dict(
    env=dict(
        map_name='infestor_viper',
        difficulty=3,
        mirror_opponent=False,
        collector_env_num=8,
        evaluator_env_num=1,
        stop_value=0.999,
        n_evaluator_episode=4,
        manager=dict(
            reset_timeout=6000,
            connect_timeout=6000,
            shared_memory=False,
        )
    ),
    policy=dict(
        cuda=True,
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=2,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, obs_shape=404.
            # For infestor_viper, obs_shape=109
            obs_shape=109,
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, global_obs_shape=216; for 2c_vs_64zg, global_obs_shape=342.
            # For infestor_viper, global_obs_shape=87
            global_obs_shape=87,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, action_shape=14 (6+8); for 2c_vs_64zg, action_shape=70 (6+64).
            # For infestor_viper, action_shape=15
            action_shape=15,
            # (List[int]) The size of hidden layer
            hidden_size_list=[
                64, 128, 128,
            ],
            # (bool) Whether to use mixer for q_value aggregation
            mixer=True
        ),
        agent_num=2,
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            target_update_theta=0.001,
            discount_factor=0.99,
        ),
        collect=dict(
            n_episode=16,
            unroll_len=16,
            env_num=8,
        ),
        eval=dict(
            env_num=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=200000,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
            )
        )
    ),
)
smac_qmix_config = EasyDict(smac_qmix_config)
main_config = smac_qmix_config
smac_qmix_create_config = dict(
    env=dict(
        type='smac',
        import_names=['app_zoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qmix'),
)
smac_qmix_create_config = EasyDict(smac_qmix_create_config)
create_config = smac_qmix_create_config
