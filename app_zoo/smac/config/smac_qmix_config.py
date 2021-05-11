from easydict import EasyDict

agent_num = 8

smac_qmix_config = dict(
    env=dict(
        map_name='3s5z',
        difficulty=8,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        # collector_env_num=16,
        # evaluator_env_num=10,
        collector_env_num=1,
        evaluator_env_num=1,
        shared_memory=False,
        stop_value=0.999,
        # n_episode=30,
        n_episode=2,
    ),
    policy=dict(
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            #agent_num=agent_num,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            obs_shape=150,
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
            global_obs_shape=216,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
            action_shape=14,
            # (List[int]) The size of hidden layer
            hidden_size_list=[
                64,
            ],
            # (bool) Whether to use mixer for q_value aggregation
            mixer=True
        ),
    ),
)
smac_qmix_config = EasyDict(smac_qmix_config)
main_config = smac_qmix_config
smac_qmix_create_config = dict(
    env=dict(
        type='smac',
        import_names=['app_zoo.smac.envs.smac_env'],
    ),
    env_manager=dict(
        type='subprocess'
    ),
    policy=dict(type='qmix'),
)
smac_qmix_create_config = EasyDict(smac_qmix_create_config)
create_config = smac_qmix_create_config
