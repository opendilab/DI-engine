from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8

main_config = dict(
    exp_name='multi_mujoco_ant_2x4_ppo',
    env=dict(
        scenario='Ant-v2',
        agent_conf="2x4d",
        agent_obsk=2,
        add_agent_id=False,
        episode_limit=1000,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='continuous',
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=2,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
            agent_obs_shape=54,
            #global_obs_shape=216,
            global_obs_shape=111,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
            action_shape=4,
            # (List[int]) The size of hidden layer
            # hidden_size_list=[64],
            action_space='continuous'
        ),
        # used in state_num of hidden_state
        learn=dict(
            epoch_per_collect=3,
            batch_size=800,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.001,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=5,
        ),
        collect=dict(env_num=collector_env_num, n_sample=3200),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=1000, )),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='mujoco_multi',
        import_names=['dizoo.multiagent_mujoco.envs.multi_mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0, max_env_step=int(1e7))
