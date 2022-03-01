from easydict import EasyDict

league_demo_ppo_config = dict(
    exp_name="slime_volley_league_ppo",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=1000000,
        env_id="SlimeVolley-v0",
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=12,
            action_shape=6,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64],
            critic_head_hidden_size=64,
            actor_head_hidden_size=64,
            share_encoder=False,
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_episode=32,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        other=dict(
            league=dict(
                player_category=['default'],
                path_policy="slime_volley_league_ppo/policy",
                active_players=dict(main_player=1,
                                    # main_exploiter=1,
                                    # league_exploiter=1,
                                    ),
                main_player=dict(
                    one_phase_step=2000,
                    branch_probs=dict(pfsp=1.0, ),
                    strong_win_rate=0.7,
                ),
                main_exploiter=dict(
                    one_phase_step=2000,
                    branch_probs=dict(main_players=1.0, ),
                    strong_win_rate=0.7,
                    min_valid_win_rate=0.3,
                ),
                league_exploiter=dict(
                    one_phase_step=2000,
                    branch_probs=dict(pfsp=1.0, ),
                    strong_win_rate=0.7,
                    mutate_prob=0.5,
                ),
                use_pretrain=False,
                use_pretrain_init_historical=False,
                payoff=dict(
                    type='battle',
                    decay=0.99,
                    min_win_rate_games=4,
                ),
                metric=dict(
                    mu=0,
                    sigma=25 / 3,
                    beta=25 / 3 / 2,
                    tau=0.0,
                    draw_probability=0.02,
                ),
            ),
        ),
    ),
)
main_config = EasyDict(league_demo_ppo_config)
