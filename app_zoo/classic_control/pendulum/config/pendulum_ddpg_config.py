from easydict import EasyDict

pendulum_ddpg_config = dict(
    seed=0,
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        # (bool) Scale output action into legal range.
        use_act_scale=True,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        model=dict(
            obs_dim=3,
            action_dim=1,
            twin_critic=False,
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            actor_update_freq=1,
            use_noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=48,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=20000,
            max_use=16,
        ), ),
    ),
)
pendulum_ddpg_config = EasyDict(pendulum_ddpg_config)
main_config = pendulum_ddpg_config

pendulum_ddpg_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['app_zoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ddpg'),
)
pendulum_ddpg_create_config = EasyDict(pendulum_ddpg_create_config)
create_config = pendulum_ddpg_create_config
