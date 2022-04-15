from easydict import EasyDict

selfplay_demo_ppo_config = dict(
    exp_name="selfplay_demo_ppo",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=100,
        env_type='prisoner_dilemma',  # ['zero_sum', 'prisoner_dilemma']
        stop_value=[-10.1, -5.05],  # prisoner_dilemma
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        model=dict(
            obs_shape=2,
            action_shape=2,
            action_space='discrete',
            encoder_hidden_size_list=[32, 32],
            critic_head_hidden_size=32,
            actor_head_hidden_size=32,
            share_encoder=False,
        ),
        learn=dict(
            update_per_collect=3,
            batch_size=32,
            learning_rate=0.00001,
            entropy_weight=0.0,
        ),
        collect=dict(
            n_episode=128, unroll_len=1, discount_factor=1.0, gae_lambda=1.0, collector=dict(get_train_sample=True, )
        ),
    ),
)
selfplay_demo_ppo_config = EasyDict(selfplay_demo_ppo_config)
# This config file can be executed by `dizoo/league_demo/selfplay_demo_ppo_main.py`
