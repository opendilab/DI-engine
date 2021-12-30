from easydict import EasyDict

cartpole_gcl_ppo_onpolicy_config = dict(
    exp_name='cartpole_guided_cost',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=5,
        batch_size=32,
        continuous=False,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=False,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=4,
            action_shape=2,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            demonstration_info_path='path',
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=50,
                cfg_type='InteractionSerialEvaluatorDict',
                stop_value=195,
                n_episode=5,
            ),
        ),
    ),
)
cartpole_gcl_ppo_onpolicy_config = EasyDict(cartpole_gcl_ppo_onpolicy_config)
main_config = cartpole_gcl_ppo_onpolicy_config
cartpole_gcl_ppo_onpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
    reward_model=dict(type='guided_cost'),
)
cartpole_gcl_ppo_onpolicy_create_config = EasyDict(cartpole_gcl_ppo_onpolicy_create_config)
create_config = cartpole_gcl_ppo_onpolicy_create_config
