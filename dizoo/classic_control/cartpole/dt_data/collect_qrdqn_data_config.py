from easydict import EasyDict

main_config = dict(
    exp_name='cartpole',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=1,
            retry_type='reset',
            auto_reset=True,
            step_timeout=None,
            reset_timeout=None,
            retry_waiting_time=0.1,
            cfg_type='BaseEnvManagerDict',
            type='base',
        ),
        # collector_env_num=8,
        # evaluator_env_num=5,
        # n_evaluator_episode=5,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=64,
        ),
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(num_workers=0, ),
                log_policy=True,
                hook=dict(
                    # load_ckpt_before_run='./cartpole/ckpt/ckpt_best.pth.tar',
                    load_ckpt_before_run=
                    '/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/ckpt/ckpt_best.pth.tar',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=False,
                ),
                cfg_type='BaseLearnerDict',
                # load_path='./cartpole/ckpt/ckpt_best.pth.tar',
                load_path=
                '/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/ckpt/ckpt_best.pth.tar',
            ),
            multi_gpu=False,
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
            ignore_done=False,
            kappa=1.0,
        ),
        collect=dict(
            collector=dict(
                deepcopy_obs=False,
                transform_obs=False,
                collect_print_freq=100,
                cfg_type='SampleSerialCollectorDict',
                type='sample',
            ),
            unroll_len=1,
            n_sample=80,
            # save
            # data_type='hdf5',
            data_type='naive',  # TODO
            save_path=
            '/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/cartpole/qrdqn_data_10eps.pkl',
            # load
            data_path='/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/cartpole/qrdqn_data.pkl',
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                cfg_type='InteractionSerialEvaluatorDict',
                stop_value=195,
                n_episode=5,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                type='advanced',
                # replay_buffer_size=100000,
                replay_buffer_size=10,  #TODO(pu)
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
        cuda=False,
        on_policy=False,
        priority=True,
        discount_factor=0.97,
        nstep=3,
        cfg_type='QRDQNCommandModePolicyDict',
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
        type='base',
    ),
    policy=dict(type='qrdqn'),
)
create_config = EasyDict(create_config)
create_config = create_config
