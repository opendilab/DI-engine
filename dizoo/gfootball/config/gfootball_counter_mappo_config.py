from easydict import EasyDict

agent_num = 4
obs_dim = 34
collector_env_num = 8
evaluator_env_num = 32

main_config = dict(
    exp_name='gfootball_counter_mappo_seed0',
    env=dict(
        env_name='academy_counterattack_hard',
        agent_num=agent_num,
        obs_dim=obs_dim,
        n_evaluator_episode=32,
        stop_value=1,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        # share_weight=True,
        multi_agent=True,
        model=dict(
            # (int) agent_num: The number of the agent.
            agent_num=agent_num,
            # (int) obs_shape: The shape of observation of each agent.
            # (int) global_obs_shape: The shape of global observation.
            agent_obs_shape=obs_dim,
            global_obs_shape=int(obs_dim * 2),
            # (int) action_shape: The number of action which each agent can take.
            action_shape=19,
        ),
        learn=dict(
            epoch_per_collect=10,
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
            clip_ratio=0.05,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(env_num=collector_env_num, n_sample=3200),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=50, )),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='gfootball-academy',
        import_names=['dizoo.gfootball.envs.gfootball_academy_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c gfootball_counter_mappo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
