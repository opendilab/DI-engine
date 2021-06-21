from easydict import EasyDict

reacher_sac_default_config = dict(
    env=dict(
        env_id='Reacher-v2',
        norm_obs=dict(use_norm=True, ),
        norm_reward=dict(
            use_norm=False,
            reward_discount=0.98,
        ),
        env_manager_type='subprocess',
        import_names=['app_zoo.mujoco.envs.mujoco_env'],
        env_type='mujoco',
        collector_env_num=16,
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
            obs_dim=11,
            action_dim=2,
            use_twin_q=True,
        ),
        learn=dict(
            train_iteration=1,
            batch_size=256,
            learning_rate_q=0.0003,
            learning_rate_value=0.0003,
            learning_rate_policy=0.0003,
            learning_rate_alpha=0.003,
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
            unroll_len=1,
            algo=dict(noise_sigma=0.1, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=100000,
            max_use=256,
        ),
    ),
    collector=dict(
        n_sample=64,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=1000,
        stop_value=-3.1,
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
reacher_sac_default_config = EasyDict(reacher_sac_default_config)
main_config = reacher_sac_default_config
