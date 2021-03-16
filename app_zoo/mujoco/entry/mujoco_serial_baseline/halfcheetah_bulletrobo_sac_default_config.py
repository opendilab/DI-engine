from easydict import EasyDict

halfcheetah_sac_default_config = dict(
    env=dict(
        # env_id='HalfCheetah-v3',  # Original MuJoCo
        # env_id='HalfCheetahMuJoCoEnv-v0',  # PyBullet MuJoCo
        env_id='HalfCheetahPyBulletEnv-v0',  # PyBullet RboSchool
        norm_obs=dict(use_norm=True, ),
        norm_reward=dict(
            use_norm=False,
            reward_discount=0.98,
        ),
        env_manager_type='subprocess',
        import_names=['app_zoo.mujoco.envs.mujoco_env'],
        env_type='mujoco',
        actor_env_num=16,
        evaluator_env_num=8,
        use_act_scale=True,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='sac',
        import_names=['nervex.policy.sac'],
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=17,
            action_dim=6,
            use_twin_q=True,
        ),
        learn=dict(
            train_step=4,
            batch_size=256,
            learning_rate_q=0.0003,
            learning_rate_value=0.0003,
            learning_rate_policy=0.0003,
            learning_rate_alpha=0.003,
            weight_decay=0.0001,
            ignore_done=True,
            algo=dict(
                target_theta=0.005,
                discount_factor=0.99,
                use_twin_q=True,
                alpha=0.2,
                reparameterization=True,
                policy_std_reg_weight=0.001,
                policy_mean_reg_weight=0.001,
                is_auto_alpha=True,
            ),
            init_data_count=5000,
        ),
        collect=dict(
            traj_len=1,
            unroll_len=1,
            algo=dict(noise_sigma=0.1, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=1000000,
            max_reuse=16,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=64,
        traj_len=1,
        traj_print_freq=1000,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=1000,
        stop_val=11000,
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
halfcheetah_sac_default_config = EasyDict(halfcheetah_sac_default_config)
main_config = halfcheetah_sac_default_config
