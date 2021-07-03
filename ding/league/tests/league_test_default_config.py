from easydict import EasyDict

league_test_config = dict(
    league=dict(
        # league_type='fake',
        import_names=['ding.league'],
        # ---player----
        player_category=['zerg', 'terran', 'protoss'],
        active_players=dict(
            main_player=1,
            main_exploiter=1,
            league_exploiter=2,
        ),
        main_player=dict(
            branch_probs=dict(
                pfsp=0.5,
                sp=0.35,
                verification=0.15,
            ),
            strong_win_rate=0.7,
            one_phase_step=2000,
        ),
        main_exploiter=dict(
            branch_probs=dict(main_players=1.0, ),
            strong_win_rate=0.7,
            one_phase_step=2000,
            min_valid_win_rate=0.2,
        ),
        league_exploiter=dict(
            branch_probs=dict(pfsp=1.0, ),
            strong_win_rate=0.7,
            one_phase_step=2000,
            mutate_prob=0.25,
        ),
        # solo_active_player:
        #     one_phase_step=2000
        #     forward_kwargs:
        #         exploration_type=[]
        #     env_kwargs:
        #         env_num=8
        #         episode_num=2
        #     adder_kwargs:
        #         use_gae=False
        #         data_push_length=128
        #     job:
        #         agent_update_freq=30  # second
        #         compressor='none'
        use_pretrain=True,
        use_pretrain_init_historical=True,
        pretrain_checkpoint_path=dict(
            zerg='pretrain_checkpoint_zerg.pth',
            terran='pretrain_checkpoint_terran.pth',
            protoss='pretrain_checkpoint_protoss.pth',
        ),
        # ---payoff---
        payoff=dict(
            type='battle',
            decay=0.99,
            min_win_rate_games=8,
        ),
    ),
)
league_test_config = EasyDict(league_test_config)
