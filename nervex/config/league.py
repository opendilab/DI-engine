from easydict import EasyDict

solo_league_default_config = dict(
    league=dict(
        league_type='solo',
        import_names=["nervex.league"],
        # ---player----
        # Just a name. Depends on the env.
        # For example, in StarCraft, this can be ['zerg', 'terran', 'protoss'].
        player_category=['default'],
        # Support different types of active players for solo and battle league.
        # For solo league, supports ['solo_active_player'].
        # For battle league, supports ['main player', 'main_exploiter', 'league_exploiter'].
        active_players=dict(
            solo_active_player=1,  # {player_type: player_num}
        ),
        solo_active_player=dict(
            # For StarCraft players, there should be keys ['branch_probs', 'strong_win_rate'].
            # Specifically for 'main_exploiter', there should be an additional key ['min_valid_win_rate'].
            one_phase_step=2e3,
            # forward_kwargs:
            #     exploration_type: ['eps']  # ['eps', 'gauss_noise', 'ou_noise']
            #     eps_kwargs:
            #         start: 0.95
            #         end: 0.05
            #         decay_len: 10000
            # adder_kwargs:
            #     use_gae: False
            #     data_push_length: 128
            env_kwargs=dict(
                env_num=8,
                episode_num=2,
            ),
            job=dict(
                agent_update_freq=30,  # second
                compressor='none',
            ),
        ),
        # "use_pretrain" means whether to use pretrain model to initialize active player.
        use_pretrain=False,
        # "use_pretrain_init_historical" means whether to use pretrain model to initialize historical player.
        # If False, "pretrain_checkpoint_path" can be omitted as well, and there will be no initial historical player;
        # Otherwise, "pretrain_checkpoint_path" should list paths of all player categories.
        use_pretrain_init_historical=False,
        pretrain_checkpoint_path=dict(solo='pretrain_checkpoint_solo.pth', ),
        # ---payoff---
        payoff=dict(
            # Supports ['solo', 'battle']
            type='solo',
            # For solo payoff, there should be ['buffer_size']
            buffer_size=4,
            # For battle payoff, there should be ['decay', 'min_win_rate_games']
            # decay=0.99,
            # min_win_rate_games=8,
        ),
        # ---runner---
        max_active_player_job=3,
        # Used for threads to sleep.
        time_interval=1,
    ),
)
solo_league_default_config = EasyDict(solo_league_default_config)
