from easydict import EasyDict
from torch.nn.modules.activation import Threshold

league_demo_ppo_config = dict(
    exp_name="league_demo_ppo",
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
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            scheduler=dict(
                schedule_flag=False,
                schedule_mode='reduce',
                factor=0.005,
                change_range=[0, 1],
                threshold=0.5,
                patience=50,
                # cooldown=0,
            ),
            learner=dict(log_policy=False),
        ),
        collect=dict(
            n_episode=128, unroll_len=1, discount_factor=1.0, gae_lambda=1.0, collector=dict(get_train_sample=True, )
        ),
        other=dict(
            league=dict(
                player_category=['default'],
                path_policy="league_demo_ppo/policy",
                active_players=dict(
                    main_player=1,
                    main_exploiter=1,
                    league_exploiter=1,
                ),
                main_player=dict(
                    one_phase_step=200,
                    branch_probs=dict(
                        pfsp=0.5,
                        sp=0.5,
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
                    mutate_prob=0.5,
                ),
                use_pretrain=False,
                use_pretrain_init_historical=False,
                payoff=dict(
                    type='battle',
                    decay=0.99,
                    min_win_rate_games=8,
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
league_demo_ppo_config = EasyDict(league_demo_ppo_config)
league_demo_ppo_create_config = EasyDict({})
