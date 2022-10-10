from easydict import EasyDict

ant_trex_ppo_config = dict(
    exp_name='ant_trex_onppo_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        env_id='Ant-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=6000,
    ),
    reward_model=dict(
        type='trex',
        min_snippet_length=10,
        max_snippet_length=100,
        checkpoint_min=100,
        checkpoint_max=900,
        checkpoint_step=100,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='abs_data_path + ./ant.params',
        continuous=True,
        # Path to the offline dataset
        # See ding/entry/application_entry_trex_collect_data.py to collect the data
        offline_data_path='abs_data_path',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=111,
            action_shape=8,
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
ant_trex_ppo_config = EasyDict(ant_trex_ppo_config)
main_config = ant_trex_ppo_config

ant_trex_ppo_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
ant_trex_ppo_create_config = EasyDict(ant_trex_ppo_create_config)
create_config = ant_trex_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c ant_trex_onppo_config.py -s 0`
    from ding.entry import serial_pipeline_trex_onpolicy
    serial_pipeline_trex_onpolicy((main_config, create_config), seed=0)
