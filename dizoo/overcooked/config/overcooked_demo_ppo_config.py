from easydict import EasyDict

overcooked_league_demo_ppo_config = dict(
    exp_name="overcooked_league_demo_ppo",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=10,
        stop_value=80,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=[5, 4, 26],
            action_shape=6,
            share_encoder=False,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=4,
            batch_size=128,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_episode=8, unroll_len=1, discount_factor=0.99, gae_lambda=0.95, collector=dict(get_train_sample=True, )
        ),
        other=dict(
            league=dict(
                player_category=['default'],
                path_policy="league_demo_ppo/policy",
                active_players=dict(
                    main_player=1,
                    main_exploiter=1,
                    #league_exploiter=1,
                ),
                main_player=dict(
                    one_phase_step=500,
                    branch_probs=dict(
                        # pfsp=0.5,
                        sp=1.0,
                    ),
                    strong_win_rate=0.7,
                ),
                main_exploiter=dict(
                    one_phase_step=500,
                    branch_probs=dict(main_players=1.0, ),
                    strong_win_rate=0.7,
                    min_valid_win_rate=0.3,
                ),
                league_exploiter=dict(
                    one_phase_step=500,
                    branch_probs=dict(pfsp=0.5, ),
                    strong_win_rate=0.7,
                    mutate_prob=0.0,
                ),
                use_pretrain=False,
                use_pretrain_init_historical=False,
                payoff=dict(
                    type='battle',
                    decay=0.99,
                    min_win_rate_games=8,
                )
            ),
        ),
    ),
)
overcooked_demo_ppo_config = EasyDict(overcooked_league_demo_ppo_config)
