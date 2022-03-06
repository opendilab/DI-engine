from easydict import EasyDict

cartpole_trex_ppo_offpolicy_config = dict(
    exp_name='cartpole_trex_offppo',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='ppo',
        env_id='CartPole-v0',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=1000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./cartpole.params',
        offline_data_path='abs data path',
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
            critic_head_layer_num=1,
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
cartpole_trex_ppo_offpolicy_config = EasyDict(cartpole_trex_ppo_offpolicy_config)
main_config = cartpole_trex_ppo_offpolicy_config
cartpole_trex_ppo_offpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
cartpole_trex_ppo_offpolicy_create_config = EasyDict(cartpole_trex_ppo_offpolicy_create_config)
create_config = cartpole_trex_ppo_offpolicy_create_config
