from easydict import EasyDict

pendulum_sac_config = dict(
    seed=0,
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
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            actor_head_type='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            is_auto_alpha=False,
            value_network=False,
        ),
        collect=dict(
            n_sample=128,
            noise_sigma=0.2,
        ),
        eval=dict(
            evaluator=dict(eval_freq=100, )
        ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=100000,
        ), ),
    ),
)
pendulum_sac_config = EasyDict(pendulum_sac_config)
main_config = pendulum_sac_config

pendulum_sac_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['app_zoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac'),
)
pendulum_sac_create_config = EasyDict(pendulum_sac_create_config)
create_config = pendulum_sac_create_config
