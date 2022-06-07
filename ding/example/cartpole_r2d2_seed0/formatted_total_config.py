from easydict import EasyDict

main_config = dict(
    exp_name='cartpole_r2d2_seed0',
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
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=195,
    ),
    policy=dict(
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(num_workers=0, ),
                log_policy=True,
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True,
                ),
                cfg_type='BaseLearnerDict',
            ),
            multi_gpu=False,
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
            value_rescale=True,
            ignore_done=False,
        ),
        collect=dict(
            collector=dict(
                deepcopy_obs=False,
                transform_obs=False,
                collect_print_freq=100,
                cfg_type='SampleSerialCollectorDict',
                type='sample',
            ),
            n_sample=32,
            traj_len_inf=True,
            env_num=8,
            unroll_len=42,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=20,
                render={
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                cfg_type='InteractionSerialEvaluatorDict',
                n_episode=8,
                stop_value=195,
            ),
            env_num=8,
        ),
        other=dict(
            replay_buffer=dict(
                type='advanced',
                replay_buffer_size=100000,
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
        priority=False,
        priority_IS_weight=False,
        discount_factor=0.995,
        nstep=5,
        burnin_step=2,
        learn_unroll_len=40,
        cfg_type='R2D2PolicyDict',
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
    policy=dict(type='r2d2'),
)
create_config = EasyDict(create_config)
create_config = create_config
