from easydict import EasyDict

nstep = 3
lunarlander_dqn_config = dict(
    exp_name='lunarlander',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            # NOTE
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(num_workers=0, ),
                log_policy=True,
                hook=dict(
                    load_ckpt_before_run=
                    './ckpt_best.pth.tar',  # TODO: syspath modeified in other place, have to use abs path. May be fix in next version.
                    # load_ckpt_before_run='DI-engine/dizoo/box2d/lunarlander/dt_data/ckpt/ckpt_best.pth.tar',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=False,
                ),
                cfg_type='BaseLearnerDict',
                load_path='./ckpt_best.pth.tar',  # TODO: same like last path.
                # load_path='DI-engine/dizoo/box2d/lunarlander/dt_data/ckpt/ckpt_best.pth.tar',
            ),
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # NOTE
            # save
            # data_type='hdf5',
            data_type='naive',
            save_path='./dt_data/dqn_data_1000eps.pkl',  # TODO(pu)
            # load
            data_path='./dt_data/dqn_data_10eps.pkl',  # TODO(pu)
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            # NOTE
            replay_buffer=dict(
                type='advanced',
                # replay_buffer_size=100000,
                replay_buffer_size=1000,  # TODO(pu)
                max_use=float('inf'),
                max_staleness=float('inf'),
                alpha=0.6,
                beta=0.4,
                anneal_step=100000,
                enable_track_used_data=False,
                deepcopy=False,
                thruput_controller=dict(
                    push_sample_rate_limit=dict(
                        max=float('inf'),
                        min=0,
                    ),
                    window_seconds=30,
                    sample_min_limit_ratio=1,
                ),
                monitor=dict(
                    sampled_data_attr=dict(
                        average_range=5,
                        print_freq=200,
                    ),
                    periodic_thruput=dict(seconds=60, ),
                ),
                cfg_type='AdvancedReplayBufferDict',
            ),
        ),
    ),
)
lunarlander_dqn_config = EasyDict(lunarlander_dqn_config)
main_config = lunarlander_dqn_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config
