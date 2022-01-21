from easydict import EasyDict

pendulum_sac_data_genearation_default_config = dict(
    seed=0,
    env=dict(
        collector_env_num=10,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=10000,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
            learner=dict(
                load_path='./sac/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='./sac/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            save_path='./sac/expert.pkl',
            data_type='hdf5',
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    ),
)

pendulum_sac_data_genearation_default_config = EasyDict(pendulum_sac_data_genearation_default_config)
main_config = pendulum_sac_data_genearation_default_config

pendulum_sac_data_genearation_default_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
pendulum_sac_data_genearation_default_create_config = EasyDict(pendulum_sac_data_genearation_default_create_config)
create_config = pendulum_sac_data_genearation_default_create_config
