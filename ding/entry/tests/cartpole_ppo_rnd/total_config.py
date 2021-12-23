exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': 60,
            'auto_reset': True,
            'reset_timeout': 60,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base'
        },
        'type': 'cartpole',
        'import_names': ['dizoo.classic_control.cartpole.envs.cartpole_env'],
        'collector_env_num': 8,
        'evaluator_env_num': 5,
        'n_evaluator_episode': 5,
        'stop_value': 195
    },
    'policy': {
        'model': {
            'obs_shape': 4,
            'action_shape': 2,
            'encoder_hidden_size_list': [64, 64, 128],
            'critic_head_hidden_size': 128,
            'actor_head_hidden_size': 128
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 6,
            'batch_size': 64,
            'learning_rate': 0.001,
            'value_weight': 0.5,
            'entropy_weight': 0.01,
            'clip_ratio': 0.2,
            'adv_norm': False,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'discount_factor': 0.9,
            'gae_lambda': 0.95,
            'n_sample': 128
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 195,
                'n_episode': 5
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 10000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'type': 'ppo_offpolicy_command',
        'cuda': False,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'nstep_return': False,
        'nstep': 3,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOOffCommandModePolicyDict'
    },
    'reward_model': {
        'type': 'rnd',
        'intrinsic_reward_type': 'add',
        'learning_rate': 0.001,
        'batch_size': 32,
        'hidden_size_list': [64, 64, 128],
        'update_per_collect': 10,
        'cfg_type': 'RndRewardModelDict',
        'obs_shape': 4
    },
    'exp_name': 'cartpole_ppo_rnd',
    'seed': 0
}
