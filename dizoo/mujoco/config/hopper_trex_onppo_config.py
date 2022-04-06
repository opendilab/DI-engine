from easydict import EasyDict

hopper_trex_onppo_config = dict(
    exp_name='hopper_trex_onppo_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='ppo',
        env_id='Hopper-v3',
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=1000,
        checkpoint_max=9000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='abs_data_path + ./hopper.params',
        continuous=True,
        # Path to the offline dataset
        # See ding/entry/application_entry_trex_collect_data.py to collect the data
        offline_data_path='abs_data_path',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            action_space='continuous',
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
hopper_trex_onppo_config = EasyDict(hopper_trex_onppo_config)
main_config = hopper_trex_onppo_config

hopper_trex_onppo_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
hopper_trex_onppo_create_config = EasyDict(hopper_trex_onppo_create_config)
create_config = hopper_trex_onppo_create_config


if __name__ == "__main__":
    # or you can enter ding -m serial_trex_onpolicy -c hopper_trex_onppo_config.py -s 0
    from ding.entry import serial_pipeline_reward_model_trex_onpolicy
    serial_pipeline_reward_model_trex_onpolicy([main_config, create_config])