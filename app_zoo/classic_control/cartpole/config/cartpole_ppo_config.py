from easydict import EasyDict

cartpole_ppo_config = dict(
    seed=0,
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            embedding_size=64,
        ),
        learn=dict(
            update_per_collect=5,
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
    ),
)
cartpole_ppo_config = EasyDict(cartpole_ppo_config)
main_config = cartpole_ppo_config
cartpole_ppo_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(
        type='base'
    ),
    policy=dict(
        type='ppo'
    ),
)
cartpole_ppo_create_config = EasyDict(cartpole_ppo_create_config)
create_config = cartpole_ppo_create_config
