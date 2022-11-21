from easydict import EasyDict

hopper_td3_bc_config = dict(
    exp_name='hopper_td3_bc_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(
            use_norm=True,
            offline_stats=dict(use_offline_stats=True, ),
        ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
            alpha=2.5,
        ),
        collect=dict(
            unroll_len=1,
            noise_sigma=0.1,
            data_type='hdf5',
            # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
            # Absolute path is recommended.
            # In DI-engine, it is usually located in ``exp_name`` directory
            data_path='data_path_placeholder',
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

hopper_td3_bc_config = EasyDict(hopper_td3_bc_config)
main_config = hopper_td3_bc_config

hopper_td3_bc_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='td3_bc',
        import_names=['ding.policy.td3_bc'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_td3_bc_create_config = EasyDict(hopper_td3_bc_create_config)
create_config = hopper_td3_bc_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c hopper_td3_bc_config.py -s 0`
#     from ding.entry import serial_pipeline
#     serial_pipeline([main_config, create_config], seed=0)
