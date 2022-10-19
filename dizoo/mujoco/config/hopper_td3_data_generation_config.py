from easydict import EasyDict

hopper_td3_data_generation_config = dict(
    exp_name='hopper_td3_data_generation_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=11000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
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
            noise_sigma=0.1,
            # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
            # Absolute path is recommended.
            # In DI-engine, it is usually located in ``exp_name`` directory
            save_path='data_path_placeholder',
            data_type='hdf5',
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)

hopper_td3_data_generation_config = EasyDict(hopper_td3_data_generation_config)
main_config = hopper_td3_data_generation_config

hopper_td3_data_generation_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='td3',
        import_names=['ding.policy.td3'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_td3_data_generation_create_config = EasyDict(hopper_td3_data_generation_create_config)
create_config = hopper_td3_data_generation_create_config
