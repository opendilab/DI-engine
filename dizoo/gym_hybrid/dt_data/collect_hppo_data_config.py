from easydict import EasyDict

gym_hybrid_hppo_config = dict(
    exp_name='gym_hybrid_hppo_collect_data_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range, usually [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1e6,
        # save_replay_gif=True,
        replay_path_gif='/mnt/nfs/renjiyuan/test_file/hybrid_hppo_replay/real_hybrid/video',
        replay_path='/mnt/nfs/renjiyuan/test_file/hybrid_hppo_replay/real_hybrid/video',
    ),
    policy=dict(
        cuda=True,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.03,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(num_workers=0, ),
                log_policy=True,
                hook=dict(
                    # load_ckpt_before_run='./lunarlander/ckpt/ckpt_best.pth.tar',
                    load_ckpt_before_run='/mnt/nfs/renjiyuan/gym-hybrid-hppo-ag-result-1031/gym_hybrid_hppo_seed0_ag/ckpt/ckpt_best.pth.tar',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=False,
                ),
                cfg_type='BaseLearnerDict',
                # load_path='./cartpole/ckpt/ckpt_best.pth.tar',
                load_path='/mnt/nfs/renjiyuan/gym-hybrid-hppo-ag-result-1031/gym_hybrid_hppo_seed0_ag/ckpt/ckpt_best.pth.tar',
            )
        ),
        collect=dict(
            data_type='naive',
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
            save_path='/mnt/nfs/renjiyuan/test_file/hybrid_hppo_replay/real_hybrid/ppo_data_best_eps_seed1.pkl',  # TODO(pu)
            # load
            data_path='/mnt/nfs/renjiyuan/test_file/hybrid_hppo_replay/real_hybrid/ppo_data_best_eps_seed1.pkl',  # TODO(pu)
        ),
        eval=dict(evaluator=dict(eval_freq=200, n_episode=5), ),
        other=dict(
            # Epsilon greedy with decay.
            # eps=dict(
            #     # Decay type. Support ['exp', 'linear'].
            #     type='exp',
            #     start=0.95,
            #     end=0.1,
            #     decay=50000,
            # ),
            # NOTE
            replay_buffer=dict(
                type='advanced',
                # replay_buffer_size=100000,
                replay_buffer_size=1,  # TODO(pu)
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
gym_hybrid_hppo_config = EasyDict(gym_hybrid_hppo_config)
main_config = gym_hybrid_hppo_config

gym_hybrid_hppo_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
gym_hybrid_hppo_create_config = EasyDict(gym_hybrid_hppo_create_config)
create_config = gym_hybrid_hppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gym_hybrid_hppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    main_config.exp_name = "gym-hybrid-hppo-11_21/test_gym_hybrid_hppo_seed2_ag_c2d_ng"
    serial_pipeline_onpolicy([main_config, create_config], seed=2, max_env_step=int(3e6))
