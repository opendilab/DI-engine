Configuration
==============

.. toctree::
   :maxdepth: 2


Learner Config
~~~~~~~~~~~~~~~
.. code:: yaml

    data:
        train:
            batch_size: 128
            dataloader_type: 'online'  # refer to data/online/online_dataloader
    train:
        use_cuda: True  
        use_distributed: True  # use multi-GPU training
        max_iterations: 1e9
        batch_size: 128
        trajectory_len: 16
    logger:
        print_freq: 5
        save_freq: 200
        eval_freq: 1000000
        var_record_type: 'alphastar'  # refer to log_helper


League Manager Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~

以solo league manager为例：

.. code:: yaml

    league:
        league_type: 'solo' # now supports ['solo']
        import_names: [nervex.league.solo_league_manager]
        # ---player----
        player_category: ['cateA']  # just a name, depends on the env
        active_players:
            # now supports ['solo_active_player'] for solo league manager,
            # ['main player', 'main_exploiter', 'league_exploiter'] for battle league manager.
            solo_active_player: 1  # {player_type: player_num}
        # all types of players in  'active_players' should be set as keys here
        solo_active_player:
            # for starcraft players, there should be keys ['branch_probs' and 'strong_win_rate'].
            # specifically, for 'main_exploiter', there should be additional key ['min_valid_win_rate'].
            one_phase_step: 2e5
            forward_kwargs:
                # ---exploration---
                exploration:
                    start: 0.95
                    end: 0.05
                    decay_len: 10000
            env_kwargs:
                env_num: 8
                episode_num: 2
            adder_kwargs:
                use_gae: False
                data_push_length: 128
            job:
                agent_update_freq: 30  # second
                compressor: 'none'
        # if use_pretrain_init_historical is False, pretrain_checkpoint_path can be omitted;
        # otherwise, pretrain_checkpoint_path should list path of all categories of players in 'player_category'
        use_pretrain_init_historical: False
        pretrain_checkpoint_path:
            cateA: 'pretrain_checkpoint_cateA.pth'
        # ---payoff---
        payoff:
            type: 'solo'  # now supports['solo', 'battle']
            # for solo payoff, there should be ['buffer_size']
            buffer_size: 4
            # for battle payoff, there should be ['decay', 'min_win_rate_games']
            decay: 0.99
            min_win_rate_games: 8
        # ---runner---
        max_active_player_job: 3
        time_interval: 1
