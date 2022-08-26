from easydict import EasyDict

n_agent = 3
n_landmark = n_agent
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='ptz_simple_spread_masac_seed0',
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=25,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        multi_agent=True,
        # priority=True,
        # priority_IS_weight=False,
        random_collect_size=0,
        model=dict(
            agent_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
            global_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) +
            n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            action_shape=5,
            # SAC concerned
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # learning_rates
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
            ignore_down=False,
            target_entropy=-2,
        ),
        collect=dict(
            n_sample=1600,
            unroll_len=1,
            env_num=collector_env_num,
        ),
        command=dict(),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=50, ),
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)

main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac_discrete'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_masac_config = main_config
ptz_simple_spread_masac_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_entry -c ptz_simple_spread_masac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
