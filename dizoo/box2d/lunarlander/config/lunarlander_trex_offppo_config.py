from easydict import EasyDict

lunarlander_trex_ppo_config = dict(
    exp_name='lunarlander_trex_offppo',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='ppo',
        env_id='LunarLander-v2',
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=1000,
        checkpoint_max=9000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./lunarlander.params',
        offline_data_path='abs data path',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            nstep=1,
            nstep_return=False,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_trex_ppo_config = EasyDict(lunarlander_trex_ppo_config)
main_config = lunarlander_trex_ppo_config
lunarlander_trex_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_offpolicy'),
)
lunarlander_trex_ppo_create_config = EasyDict(lunarlander_trex_ppo_create_config)
create_config = lunarlander_trex_ppo_create_config
