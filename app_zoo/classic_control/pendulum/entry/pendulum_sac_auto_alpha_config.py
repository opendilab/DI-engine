from easydict import EasyDict

use_twin_q = True
pendulum_sac_auto_alpha_config = dict(
    env=dict(
        manager=dict(
            type='base',
        ),
        env_kwargs=dict(
            import_names=['app_zoo.classic_control.pendulum.envs.pendulum_env'],
            env_type='pendulum',
            actor_env_num=8,
            evaluator_env_num=8,
            use_act_scale=True,
            norm_obs=dict(use_norm=False, ),
            norm_reward=dict(use_norm=False,),
        ),
    ),
    policy=dict(
        use_cuda=False,
        policy_type='sac',
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=3,
            action_dim=1,
            use_twin_q=use_twin_q,
        ),
        learn=dict(
            train_iteration=1,
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
                use_twin_q=use_twin_q,
                alpha=0.2,
                reparameterization=True,
                policy_std_reg_weight=0.001,
                policy_mean_reg_weight=0.001,
                is_auto_alpha=True,
            ),
        ),
        collect=dict(
            traj_len=1,
            unroll_len=1,
            algo=dict(noise_sigma=0.1, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        replay_buffer_size=100000,
        max_use=256,
    ),
    actor=dict(
        n_sample=64,
        traj_len=1,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=20,
        stop_value=-250,
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
pendulum_sac_auto_alpha_config = EasyDict(pendulum_sac_auto_alpha_config)
main_config = pendulum_sac_auto_alpha_config
