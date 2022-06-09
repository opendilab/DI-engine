from easydict import EasyDict

obs_shape = [3, 100, 100]
encoder_hidden_size_list = [32, 64, 64, 50]
cartpole_swingup_curl_sac_config = dict(
    exp_name='cartpole_swingup_curl_sac_seed0',
    env=dict(
        collector_env_num=10,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        domain_name='cartpole',
        task_name='swingup',
        stop_value=1000,
        manager=dict(shared_memory=False),
    ),
    policy=dict(
        cuda=True,
        random_collect_size=1000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=encoder_hidden_size_list[-1],
            critic_head_hidden_size=encoder_hidden_size_list[-1],
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=10,
            noise_sigma=0.2,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    obs_model=dict(
        obs_shape=[3, 100, 100],
        encoder_lr=1e-3,
        w_lr=1e-4,
        encoder_hidden_size_list=encoder_hidden_size_list,
        encoder_feature_size=encoder_hidden_size_list[-1],
    )
)
cartpole_swingup_curl_sac_config = EasyDict(cartpole_swingup_curl_sac_config)
main_config = cartpole_swingup_curl_sac_config

cartpole_swingup_curl_sac_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac'),
)
cartpole_swingup_curl_sac_create_config = EasyDict(cartpole_swingup_curl_sac_create_config)
create_config = cartpole_swingup_curl_sac_create_config
