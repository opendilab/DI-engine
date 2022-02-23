from easydict import EasyDict

ant_trex_ppo_default_config = dict(
    exp_name='ant_trex_onppo',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        env_id='Ant-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=6000,
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='ppo',
        env_id='Ant-v3',
        min_snippet_length=10,
        max_snippet_length=100,
        checkpoint_min=100,
        checkpoint_max=900,
        checkpoint_step=100,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./ant.params',
        continuous=True,
        offline_data_path='asb data path',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=111,
            action_shape=8,
            continuous=True,
        ),
        continuous=True,
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
ant_trex_ppo_default_config = EasyDict(ant_trex_ppo_default_config)
main_config = ant_trex_ppo_default_config

ant_trex_ppo_create_default_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
ant_trex_ppo_create_default_config = EasyDict(ant_trex_ppo_create_default_config)
create_config = ant_trex_ppo_create_default_config
