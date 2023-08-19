from easydict import EasyDict

cartpole_balance_ppo_config = dict(
    exp_name='dmc2gym_cartpole_balance_ppo',
    env=dict(
        env_id='dmc2gym_cartpole_balance',
        domain_name='cartpole',
        task_name='balance',
        from_pixels=False,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=1000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=5,
            action_shape=1,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    )
)
cartpole_balance_ppo_config = EasyDict(cartpole_balance_ppo_config)
main_config = cartpole_balance_ppo_config

cartpole_balance_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
    replay_buffer=dict(type='naive', ),
)
cartpole_balance_create_config = EasyDict(cartpole_balance_create_config)
create_config = cartpole_balance_create_config

# To use this config, you can enter dizoo/dmc2gym/entry to call dmc2gym_onppo_main.py
