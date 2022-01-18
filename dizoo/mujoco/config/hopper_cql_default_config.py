from easydict import EasyDict

hopper_cql_default_config = dict(
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=1e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
            with_lagrange=False,
            lagrange_thresh=-1.0,
            min_q_weight=5.0,
        ),
        collect=dict(
            unroll_len=1,
            data_type='naive',
            data_path='./default_experiment/expert_iteration_200000.pkl',
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

hopper_cql_default_config = EasyDict(hopper_cql_default_config)
main_config = hopper_cql_default_config

hopper_cql_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='cql',
        import_names=['ding.policy.cql'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_cql_default_create_config = EasyDict(hopper_cql_default_create_config)
create_config = hopper_cql_default_create_config
