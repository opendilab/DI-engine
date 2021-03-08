from easydict import EasyDict

one_vs_one_league_default_config = dict(
    league=dict(
        league_type='one_vs_one',
        import_names=["nervex.league"],
        # ---player----
        # "player_category" is just a name. Depends on the env.
        # For example, in StarCraft, this can be ['zerg', 'terran', 'protoss'].
        player_category=['default'],
        # Support different types of active players for solo and battle league.
        # For solo league, supports ['solo_active_player'].
        # For battle league, supports ['battle_active_player', 'main_player', 'main_exploiter', 'league_exploiter'].
        active_players=dict(
            naive_sp_player=1,  # {player_type: player_num}
        ),
        naive_sp_player=dict(
            # There should be keys ['one_phase_step', 'branch_probs', 'strong_win_rate'].
            # Specifically for 'main_exploiter' of StarCraft, there should be an additional key ['min_valid_win_rate'].
            one_phase_step=10,
            branch_probs=dict(
                pfsp=0.5,
                sp=0.5,
            ),
            strong_win_rate=0.7,
        ),
        # "use_pretrain" means whether to use pretrain model to initialize active player.
        use_pretrain=False,
        # "use_pretrain_init_historical" means whether to use pretrain model to initialize historical player.
        # "pretrain_checkpoint_path" is the pretrain checkpoint path used in "use_pretrain" and 
        # "use_pretrain_init_historical". If both are False, "pretrain_checkpoint_path" can be omitted as well.
        # Otherwise, "pretrain_checkpoint_path" should list paths of all player categories.
        use_pretrain_init_historical=False,
        pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth', ),
        # ---payoff---
        payoff=dict(
            # Supports ['battle']
            type='battle',
            decay=0.99,
            min_win_rate_games=8,
        ),
    ),
)
one_vs_one_league_default_config = EasyDict(one_vs_one_league_default_config)
