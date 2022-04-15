from easydict import EasyDict

league_demo_ppo_config = dict(
    exp_name="slime_volley_league_ppo_seed0",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # we don't set agent_vs_agent field because it should be set in entry for different usage.
        stop_value=5,  # 5 times per episode
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
            share_encoder=False,  # It is not wise to share encoder in low-dimension observation.
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=256,
            learning_rate=3e-4,
            entropy_weight=0.001,  # [0.01, 0.001, 0.0]
            clip_ratio=0.2,
        ),
        collect=dict(
            n_episode=16,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        other=dict(
            league=dict(
                player_category=['default'],
                # path to save policy of league player, user can specify this field, such as:
                # path_policy="slime_volley_league_ppo_seed0/policy"
                path_policy="policy_path_placeholer",
                active_players=dict(main_player=1, ),
                main_player=dict(
                    one_phase_step=20000,
                    branch_probs=dict(pfsp=0.2, sp=0.8),
                    strong_win_rate=0.7,
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
# this config can be executed by two entry function for different usage
# - dizoo/slime_volley/entry/slime_volley_selfplay_ppo_main.py
# - dizoo/slime_volley/entry/slime_volley_league_ppo_main.py
