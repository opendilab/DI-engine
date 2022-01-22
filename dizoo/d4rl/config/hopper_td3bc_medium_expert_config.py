from easydict import EasyDict

main_config = dict(
    exp_name='td3-bc-train_d4rl_seed0',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=1,
            step_timeout=60,
            auto_reset=True,
            reset_timeout=60,
            retry_waiting_time=0.1,
            cfg_type='BaseEnvManagerDict',
            type='base',
        ),
        env_id='hopper-medium-expert-v0',
        norm_obs={'use_norm': False},
        norm_reward={'use_norm': False},
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        model=dict(
            twin_critic=True,
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            actor_head_type='regression',
        ),
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(num_workers=0, ),
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=10000,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True,
                ),
                cfg_type='BaseLearnerDict',
            ),
            multi_gpu=False,
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=0.0003,
            learning_rate_critic=0.0003,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range={
                'min': -0.5,
                'max': 0.5
            },
            alpha=2.5,
            normalize_states=True,
            train_epoch=30000,
            lr_scheduler={
                'flag': False,
                'T_max': 1000000,
                'type': 'Cosine'
            },
            optimizer={
                'type': 'adam',
                'weight_decay': 0
            },
            lmbda_type='q_value',
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
            noise_sigma=0.1,
            normalize_states=True,
            data_type='d4rl',
            data_path=None,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=10000,
                cfg_type='InteractionSerialEvaluatorDict',
                stop_value=6000,
                n_episode=8,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                type='naive',
                replay_buffer_size=2000000,
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
        cuda=True,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        random_collect_size=25000,
        cfg_type='TD3BCCommandModePolicyDict',
        import_names=['ding.policy.td3_bc'],
        command={},
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
        type='base',
    ),
    policy=dict(type='td3_bc'),
)
create_config = EasyDict(create_config)
create_config = create_config
