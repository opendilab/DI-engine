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
        save_path: '.'  # save ckpt/log path, defaults to the current directory '.'
        load_path: ''  # load ckpt path, if load_path == '', do not load anything
    env:
        # env manager type, base is pseudo parallel and subprocess is subprocess parallel, the former is used for some light env(e.g.: cartpole),
        # which env step time is much shorter than IPC time, and the latter is used for more complicated env(e.g.: pong)
        env_manager_type: 'base'  # [base, subprocess]
        import_names: ['app_zoo.classic_control.cartpole.envs.cartpole_env']  # the user must indicate the absolute path of env
        env_type: 'cartpole'  # env register name
        actor_env_num: 8  # the number of env used in actor for data collect
        evaluator_env_num: 5  # the number of env used in evaluator for performance metric
    policy:
        use_cuda: False  # whether use cuda for network
        policy_type: 'dqn'  # RL policy register name
        import_names: ['nervex.policy.dqn']  # the user must indicate the absolute path of policy
        on_policy: False  # whether is a on-policy RL algorithm, which means some specific operation in training loop, such as reset buffer when each training iteration ends
        model:  # model arguments, which is directly used for creating model
            obs_dim: 4
            action_dim: 2
            embedding_dim: 64
            dueling: True  # whether use dueling head
        learn:
            train_step: 3  # train step, collect data -> train fixed step -> collect data
            batch_size: 64
            learning_rate: 0.001
            weight_decay: 0.0  # L2 norm weight for network parameters
            algo:
                target_update_freq: 100  # the update iteration frequency of target Q network
                discount_factor: 0.95  # the future discount factor, usually named gamma
        collect:
            traj_len: 1  # the length of trajectory, the basic length of actor data
            unroll_len: 1  # the length of unroll, the basic length of learner training data
        command:  # command is the component for the communication between modules
            eps:
                type: 'exp'  # ['exp', 'linear']
                start: 0.95
                end: 0.1
                decay: 10000  # training iteration
    replay_buffer:
        meta_maxlen: 100000  # replay buffer max length
        max_reuse: 100  # max reuse number of a data sample
        unroll_len: 1  # policy.collect.unroll_len
        min_sample_ratio: 1  # minimum sample number->(batch_size * sample_ratio)
    actor:
        n_sample: 8  # the sample number of a execution of actor collect, a sample can include many steps 
        traj_len: 1  # policy.collect.traj_len
        traj_print_freq: 100
        collect_print_freq: 100
    evaluator:
        n_episode: 5  # the episode number of eval
        eval_freq: 200  # training iteration
        stop_val: 195  # if final_eval_reward is greater than this value, the training is converged
    learner:
        hook:
            log_show:
                name: log_show
                type: log_show
                priority: 20
                position: after_iter
                ext_args:
                    freq: 100
    command:
        placeholder: 'placeholder'


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
