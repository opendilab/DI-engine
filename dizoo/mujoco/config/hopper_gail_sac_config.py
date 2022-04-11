from easydict import EasyDict

obs_shape = 11
act_shape = 3
hopper_sac_gail_config = dict(
<<<<<<< HEAD:dizoo/mujoco/config/hopper_sac_gail_config.py
    exp_name='hopper_sac_gail_seed0',
=======
    exp_name='hopper_gail_sac_seed0',
>>>>>>> c7975302095288cc19f58bde0f9964e7fbd206a3:dizoo/mujoco/config/hopper_gail_sac_config.py
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        input_size=obs_shape + act_shape,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
<<<<<<< HEAD:dizoo/mujoco/config/hopper_sac_gail_config.py
        # Users should add their own data path here. 
        # Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        expert_data_path='data_path_placeholder',
        # state_dict of the reward model. Model path should lead to a model.
        # Absolute path is recommended.
        load_path='model_path_placeholder',
        # path to the expert state_dict.
        expert_load_path='model_path_placeholder',
=======
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='data_path_placeholder+/reward_model/ckpt/ckpt_best.pth.tar',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        data_path='data_path_placeholder',
>>>>>>> c7975302095288cc19f58bde0f9964e7fbd206a3:dizoo/mujoco/config/hopper_gail_sac_config.py
        collect_count=100000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=act_shape,
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
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

hopper_sac_gail_config = EasyDict(hopper_sac_gail_config)
main_config = hopper_sac_gail_config

hopper_sac_gail_create_config = dict(
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
    reward_model=dict(type='gail'),
)
hopper_sac_gail_create_config = EasyDict(hopper_sac_gail_create_config)
create_config = hopper_sac_gail_create_config
<<<<<<< HEAD:dizoo/mujoco/config/hopper_sac_gail_config.py


if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c hopper_sac_gail_config.py -s 0`
    from ding.entry import serial_pipeline_gail
    from hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    serial_pipeline_gail(
        [main_config, create_config], [hopper_sac_config, hopper_sac_create_config],
        max_iterations=1000000,
=======

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c hopper_gail_sac_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. hopper_sac_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.mujoco.config.hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    expert_main_config = hopper_sac_config
    expert_create_config = hopper_sac_create_config
    serial_pipeline_gail(
        [main_config, create_config], [expert_main_config, expert_create_config],
        max_env_step=1000000,
>>>>>>> c7975302095288cc19f58bde0f9964e7fbd206a3:dizoo/mujoco/config/hopper_gail_sac_config.py
        seed=0,
        collect_data=True
    )
