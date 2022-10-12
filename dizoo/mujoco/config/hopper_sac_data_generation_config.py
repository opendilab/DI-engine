from easydict import EasyDict

hopper_sac_data_generation_config = dict(
    exp_name='hopper_sac_data_generation_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=10,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            learner=dict(
                # Model path should lead to a model.
                # Absolute path is recommended.
                # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
                load_path='model_path_placeholder',
                hook=dict(
                    load_ckpt_before_run='model_path_placeholder',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
            # Absolute path is recommended.
            # In DI-engine, it is usually located in ``exp_name`` directory
            save_path='data_path_placeholder',
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

hopper_sac_data_generation_config = EasyDict(hopper_sac_data_generation_config)
main_config = hopper_sac_data_generation_config

hopper_sac_data_genearation_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_sac_data_genearation_create_config = EasyDict(hopper_sac_data_genearation_create_config)
create_config = hopper_sac_data_genearation_create_config
