from easydict import EasyDict

obs_shape = 17
act_shape = 6
halfcheetah_sac_gail_config = dict(
    exp_name='halfcheetah_sac_gail_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=12000,
    ),
    reward_model=dict(
        input_size=obs_shape + act_shape,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
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
        collect_count=300000,
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

halfcheetah_sac_gail_config = EasyDict(halfcheetah_sac_gail_config)
main_config = halfcheetah_sac_gail_config

halfcheetah_sac_gail_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_sac_gail_create_config = EasyDict(halfcheetah_sac_gail_create_config)
create_config = halfcheetah_sac_gail_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c ant_gail_sac_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. hopper_sac_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.mujoco.config.halfcheetah_sac_config import halfcheetah_sac_config, halfcheetah_sac_create_config

    expert_main_config = halfcheetah_sac_config
    expert_create_config = halfcheetah_sac_create_config
    serial_pipeline_gail(
        [main_config, create_config], [expert_main_config, expert_create_config],
        max_env_step=10000000,
        seed=0,
        collect_data=True
    )
