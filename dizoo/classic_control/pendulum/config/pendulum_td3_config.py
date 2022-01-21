from easydict import EasyDict

pendulum_td3_config = dict(
    exp_name='pendulum_td3',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=800,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.1,
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
        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
    ),
)
pendulum_td3_config = EasyDict(pendulum_td3_config)
main_config = pendulum_td3_config

pendulum_td3_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='td3'),
)
pendulum_td3_create_config = EasyDict(pendulum_td3_create_config)
create_config = pendulum_td3_create_config
