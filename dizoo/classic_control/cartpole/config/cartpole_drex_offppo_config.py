from easydict import EasyDict

cartpole_drex_ppo_offpolicy_config = dict(
    exp_name='cartpole_drex_offppo_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='drex',
        algo_for_model='ppo',
        env_id='CartPole-v0',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=1000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        # path to expert models that generate demonstration data
        # Users should add their own model path here. Model path should lead to an exp_name.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name``.
        # For example, if you want to use dqn to generate demos, you can use ``spaceinvaders_dqn``
        expert_model_path='expert_model_path_placeholder',
        # path to save reward model
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then the reward model will be saved in this directory.
        reward_model_path='reward_model_path_placeholder + ./spaceinvaders.params',
        # path to save generated observations.
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then all the generated data will be saved in this directory.
        offline_data_path='offline_data_path_placeholder',
        # path to pretrained bc model. If omitted, bc will be trained instead.
        # Users should add their own model path here. Model path should lead to a model ckpt.
        # Absolute path is recommended.
        bc_path='bc_path_placeholder',
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
cartpole_drex_ppo_offpolicy_config = EasyDict(cartpole_drex_ppo_offpolicy_config)
main_config = cartpole_drex_ppo_offpolicy_config
cartpole_drex_ppo_offpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_offpolicy'),
)
cartpole_drex_ppo_offpolicy_create_config = EasyDict(cartpole_drex_ppo_offpolicy_create_config)
create_config = cartpole_drex_ppo_offpolicy_create_config
