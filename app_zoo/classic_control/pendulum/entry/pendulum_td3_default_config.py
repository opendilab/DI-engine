from easydict import EasyDict

use_twin_critic = True
pendulum_td3_default_config = dict(
    env=dict(
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        env_manager_type='base',
        import_names=['app_zoo.classic_control.pendulum.envs.pendulum_env'],
        env_type='pendulum',
        actor_env_num=8,
        evaluator_env_num=8,
        use_act_scale=True,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='ddpg',
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=3,
            action_dim=1,
            use_twin_critic=use_twin_critic,
        ),
        learn=dict(
            train_step=1,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            weight_decay=0.0001,
            ignore_done=True,
            algo=dict(
                target_theta=0.005,
                discount_factor=0.99,
                actor_update_freq=2,
                use_twin_critic=use_twin_critic,
                use_noise=True,
                noise_sigma=0.2,
                noise_range=dict(
                    min=-0.5,
                    max=0.5,
                ),
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
        replay_buffer_size=20000,
        max_reuse=16,
    ),
    actor=dict(
        n_sample=48,
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
pendulum_td3_default_config = EasyDict(pendulum_td3_default_config)
main_config = pendulum_td3_default_config
