from easydict import EasyDict

cartpole_ppo_offpolicy_config = dict(
    exp_name='cartpole_ppo_offpolicy',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        on_policy=False,
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=6,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=5000))
    ),
)
cartpole_ppo_offpolicy_config = EasyDict(cartpole_ppo_offpolicy_config)
main_config = cartpole_ppo_offpolicy_config
cartpole_ppo_offpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
cartpole_ppo_offpolicy_create_config = EasyDict(cartpole_ppo_offpolicy_create_config)
create_config = cartpole_ppo_offpolicy_create_config
