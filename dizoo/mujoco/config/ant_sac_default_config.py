from easydict import EasyDict

ant_sac_default_config = dict(
    exp_name='ant_sac',
    env=dict(
        env_id='Ant-v3',
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
        random_collect_size=10000,
        model=dict(
            obs_shape=111,
            action_shape=8,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

ant_sac_default_config = EasyDict(ant_sac_default_config)
main_config = ant_sac_default_config

ant_sac_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
ant_sac_default_create_config = EasyDict(ant_sac_default_create_config)
create_config = ant_sac_default_create_config
