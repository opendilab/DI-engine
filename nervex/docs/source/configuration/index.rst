Configuration
==============

.. toctree::
   :maxdepth: 2

CartPole DQN Config
~~~~~~~~~~~~~~~~~~~~~


cartpole_dqn_default_config.yaml

.. code:: yaml

    common:
        name: CartpoleDqnConfig
        time_wrapper_type: cuda  # use torch.cuda.Event and torch.cuda.synchronize for time record
        save_path: '.'  # save ckpt/log path
        load_path: ''  # load ckpt path, if load_path == '', do not load anything
        algo_type: 'dqn'  # ['dqn', 'drqn', 'ppo', 'a2c', 'ddpg', 'qmix', 'coma']
    learner:
        use_cuda: False  # whether use cuda for network training
        use_distributed: False  # whether use distributed training(linklink)
        max_iterations: 10000000  # max train iterations
        train_step: 1  # train step interval, collect data -> train fixed step -> collect data
        learning_rate: 0.0001
        weight_decay: 0.0  # L2 norm for network weight
        eps:
            type: 'exp'  # ['linear', 'exp']
            start: 0.95
            end: 0.05
            decay: 10000  # training iteration
        data:
            batch_size: 64
            chunk_size: 64  # dataloader per worker load data number
            num_worker: 0
            max_reuse: 100  # max reuse number of a data sample
            buffer_length: 100000  # replay buffer length
            sample_ratio: 0.5  # the ratio of generating new data for one train step
            use_mid_pack: True  # whether pack data in the middle of a episode
        dqn:
            discount_factor: 0.99
            is_double: True  # whether use double dqn(target network)
            dueling: True  # whether use dueling dqn network architecture
        hook:  # learner function hook
            save_ckpt_after_iter:
                name: save_ckpt_after_iter
                type: save_ckpt
                priority: 40  # the lower value, the higher priority
                position: after_iter  # ['before_run', 'before_iter', 'after_iter', 'after_run']
                ext_args:
                    freq: 10000
            log_show:
                name: log_show
                type: log_show
                priority: 20
                position: after_iter
                ext_args:
                    freq: 100
    env:
        placeholder: 'placeholder'
    actor:
        env_num: 10
        episode_num: 'inf'  # inf means run resetted episode until the program is over
        print_freq: 500
    evaluator:
        env_num: 10
        episode_num: 'inf'
        total_episode_num: 10  # total evaluate episode_num
        eval_step: 1500
        stop_val: 39 # 195//5  # if episode returns are greater than this value, training is over

.. note::
   由于单机同步版本数据生成和训练是串行执行，即生成足够数量的数据后训练一定迭代数，使用者可以调节 ``train_step``, ``batch_size``, ``max_reuse``, ``buffer_length`` 这四个量来控制
   算法的训练数据情况，比如令 ``max_reuse=1``，``buffer_length = train_step * batch_size，train_step=1``，即可对应标准的 on-policy训练过程。

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
