from easydict import EasyDict

slime_volley_league_ppo_config = dict(
    exp_name="slime_volley_league_ppo",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=100,
        stop_value=0,
        # Single-agent env for evaluator; Double-agent env for collector.
        # Should be assigned True or False in code.
        is_evaluator=None,
        manager=dict(shared_memory=False, ),
        env_id="SlimeVolley-v0",
    ),
    policy=dict(
        cuda=False,
        continuous=False,
        model=dict(
            obs_shape=12,
            action_shape=6,
            encoder_hidden_size_list=[32, 32],
            critic_head_hidden_size=32,
            actor_head_hidden_size=32,
            share_encoder=False,
        ),
        learn=dict(
            update_per_collect=3,
            batch_size=32,
            learning_rate=0.00001,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_episode=128, unroll_len=1, discount_factor=1.0, gae_lambda=1.0, collector=dict(get_train_sample=True, )
        ),
        other=dict(
            league=dict(
                player_category=['default'],
                path_policy="slime_volley_league_ppo/policy",
                active_players=dict(
                    main_player=1,
                    main_exploiter=1,
                    league_exploiter=1,
                ),
                main_player=dict(
                    one_phase_step=200,
                    branch_probs=dict(
                        pfsp=0.5,
                        sp=1.0,
                    ),
                    strong_win_rate=0.7,
                ),
                main_exploiter=dict(
                    one_phase_step=200,
                    branch_probs=dict(main_players=1.0, ),
                    strong_win_rate=0.7,
                    min_valid_win_rate=0.3,
                ),
                league_exploiter=dict(
                    one_phase_step=200,
                    branch_probs=dict(pfsp=1.0, ),
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
slime_volley_league_ppo_config = EasyDict(slime_volley_league_ppo_config)
