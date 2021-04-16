from easydict import EasyDict

pendulum_ppo_default_config = dict(
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
        policy_type='ppo_vanilla',
        on_policy=True,
        use_priority=False,
        model=dict(
            continous=True,
            fixed_sigma_value=0.2,
            obs_dim=3,
            action_dim=1,
            embedding_dim=64,
        ),
        learn=dict(
            train_iteration=5,
            batch_size=64,
            learning_rate=0.0005,
            weight_decay=0.0001,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.0001,
                clip_ratio=0.2,
            ),
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=1,
            algo=dict(
                discount_factor=0.9,
                gae_lambda=0.95,
            ),
        ),
    ),
    replay_buffer=dict(
        replay_buffer_size=1000,
        max_use=16,
    ),
    actor=dict(
        n_episode=16,
        traj_len=200,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=200,
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
)
pendulum_ppo_default_config = EasyDict(pendulum_ppo_default_config)
main_config = pendulum_ppo_default_config
