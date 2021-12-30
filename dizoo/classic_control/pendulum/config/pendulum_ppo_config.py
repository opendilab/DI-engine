from easydict import EasyDict

pendulum_ppo_config = dict(
    env=dict(
        collector_env_num=1,
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        recompute_adv=True,
        model=dict(
            obs_shape=3,
            action_shape=1,
            encoder_hidden_size_list=[64, 64],
            action_space='continuous',
            actor_head_layer_num=0,
            critic_head_layer_num=0,
            sigma_type='conditioned',
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=32,
            learning_rate=3e-5,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=False,
            value_norm=True,
            ignore_done=True,
        ),
        collect=dict(
            n_sample=200,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=1.,
        ),
        eval=dict(evaluator=dict(eval_freq=200, ))
    ),
)
pendulum_ppo_config = EasyDict(pendulum_ppo_config)
main_config = pendulum_ppo_config
pendulum_ppo_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
pendulum_ppo_create_config = EasyDict(pendulum_ppo_create_config)
create_config = pendulum_ppo_create_config
