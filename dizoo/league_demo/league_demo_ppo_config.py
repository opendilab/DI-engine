from easydict import EasyDict

league_demo_ppo_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=1,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=2,
            action_shape=2,
            encoder_hidden_size_list=[32, 32],
            critic_head_hidden_size=32,
            actor_head_hidden_size=32,
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
            n_episode=64, unroll_len=1, discount_factor=0.9, gae_lambda=0.95, collector=dict(get_train_sample=True, )
        ),
    ),
)
league_demo_ppo_config = EasyDict(league_demo_ppo_config)
