from easydict import EasyDict

agent_num = 8
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

main_config = dict(
    exp_name='smac_8m9m_mappo_seed0',
    env=dict(
        map_name='8m_vs_9m',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=32,
        stop_value=0.99,
        death_mask=False,
        special_global_state=special_global_state,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        on_policy=True,
        multi_agent=True,
        continuous=False,
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=agent_num,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
            agent_obs_shape=108,
            global_obs_shape=263,
            action_shape=15,
            # (List[int]) The size of hidden layer
            # hidden_size_list=[64],
            # delete encode in code
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        # used in state_num of hidden_state
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            epoch_per_collect=5,
            batch_size=3200,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(env_num=collector_env_num, n_sample=3200),
        eval=dict(
            evaluator=dict(eval_freq=100, ),
            env_num=evaluator_env_num,
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':

    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
