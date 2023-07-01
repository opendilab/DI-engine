from easydict import EasyDict

obs_shape = 11
act_shape = 3
hopper_gail_sac_config = dict(
    exp_name='hopper_gail_sac_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        input_size=obs_shape + act_shape,
        hidden_size_list=[256],
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='hopper_sac_seed0/ckpt/ckpt_best.pth.tar',
        # Path where to store the reward model
        reward_model_path='hopper_gail_sac_seed0/reward_model/ckpt/ckpt_best.pth.tar',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        data_path='hopper_sac_seed0',
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

hopper_gail_sac_config = EasyDict(hopper_gail_sac_config)
main_config = hopper_gail_sac_config

hopper_gail_sac_create_config = dict(
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
hopper_gail_sac_create_config = EasyDict(hopper_gail_sac_create_config)
create_config = hopper_gail_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c hopper_gail_sac_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. hopper_sac_config.py
    from ding.entry import serial_pipeline_reward_model_offpolicy, collect_demo_data
    from dizoo.mujoco.config.hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    # set expert config from policy config in dizoo
    expert_cfg = (hopper_sac_config, hopper_sac_create_config)
    expert_main_config = hopper_sac_config
    expert_data_path = main_config.reward_model.data_path + '/expert_data.pkl'

    # collect expert data
    collect_demo_data(
        expert_cfg,
        seed=0,
        state_dict_path=main_config.reward_model.expert_model_path,
        expert_data_path=expert_data_path,
        collect_count=main_config.reward_model.collect_count
    )

    # train reward model
    serial_pipeline_reward_model_offpolicy((main_config, create_config))
