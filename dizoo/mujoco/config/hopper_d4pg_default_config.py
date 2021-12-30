from easydict import EasyDict

hopper_d4pg_default_config = dict(
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=3000,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        nstep=5,
        random_collect_size=25000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
            critic_head_type='categorical',
            v_min=-100,
            v_max=100,
            n_atom=51,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
hopper_d4pg_default_config = EasyDict(hopper_d4pg_default_config)
main_config = hopper_d4pg_default_config

hopper_d4pg_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='d4pg',
        import_names=['ding.policy.d4pg'],
    ),
)
hopper_d4pg_default_create_config = EasyDict(hopper_d4pg_default_create_config)
create_config = hopper_d4pg_default_create_config
