from easydict import EasyDict

distar_cfg = EasyDict(
    {
        'env': {
            'manager': {
                'episode_num': 100000,
                'max_retry': 1000,
                'retry_type': 'renew',
                'auto_reset': True,
                'step_timeout': None,
                'reset_timeout': None,
                'retry_waiting_time': 0.1,
                'cfg_type': 'BaseEnvManagerDict',
                'shared_memory': False,
                'return_original_data': True
            },
            'collector_env_num': 1,
            'evaluator_env_num': 1,
            'n_evaluator_episode': 100,
            'env_type': 'prisoner_dilemma',
            'stop_value': [-10.1, -5.05]
        },
        'policy': {
            'model': {
                'obs_shape': 2,
                'action_shape': 2,
                'action_space': 'discrete',
                'encoder_hidden_size_list': [32, 32],
                'critic_head_hidden_size': 32,
                'actor_head_hidden_size': 32,
                'share_encoder': False
            },
            'learn': {
                'learner': {
                    'train_iterations': 1000000000,
                    'dataloader': {
                        'num_workers': 0
                    },
                    'log_policy': False,
                    'hook': {
                        'load_ckpt_before_run': '',
                        'log_show_after_iter': 100,
                        'save_ckpt_after_iter': 10000,
                        'save_ckpt_after_run': True
                    },
                    'cfg_type': 'BaseLearnerDict'
                },
                'multi_gpu': False,
                'epoch_per_collect': 10,
                'batch_size': 4,
                'learning_rate': 1e-05,
                'value_weight': 0.5,
                'entropy_weight': 0.0,
                'clip_ratio': 0.2,
                'adv_norm': True,
                'value_norm': True,
                'ppo_param_init': True,
                'grad_clip_type': 'clip_norm',
                'grad_clip_value': 0.5,
                'ignore_done': False,
                'update_per_collect': 3,
                'scheduler': {
                    'schedule_flag': False,
                    'schedule_mode': 'reduce',
                    'factor': 0.005,
                    'change_range': [0, 1],
                    'threshold': 0.5,
                    'patience': 50
                }
            },
            'collect': {
                'collector': {
                    'deepcopy_obs': False,
                    'transform_obs': False,
                    'collect_print_freq': 100,
                    'get_train_sample': True,
                    'cfg_type': 'BattleEpisodeSerialCollectorDict'
                },
                'discount_factor': 1.0,
                'gae_lambda': 1.0,
                'n_episode': 1,
                'unroll_len': 16
            },
            'eval': {
                'evaluator': {
                    'eval_freq': 50,
                    'cfg_type': 'BattleInteractionSerialEvaluatorDict',
                    'stop_value': [-10.1, -5.05],
                    'n_episode': 100
                }
            },
            'other': {
                'replay_buffer': {
                    'type': 'naive',
                    'replay_buffer_size': 6,
                    'max_use': 2,
                    'deepcopy': False,
                    'enable_track_used_data': False,
                    'periodic_thruput_seconds': 60,
                    'cfg_type': 'NaiveReplayBufferDict'
                },
                'league': {
                    'player_category': ['default'],
                    'path_policy': 'league_demo/ckpt',
                    'active_players': {
                        'main_player': 1
                    },
                    'main_player': {
                        'one_phase_step': 10,  # 20
                        'branch_probs': {
                            'pfsp': 0.0,
                            'sp': 0.0,
                            'sl': 1.0
                        },
                        'strong_win_rate': 0.7
                    },
                    'main_exploiter': {
                        'one_phase_step': 200,
                        'branch_probs': {
                            'main_players': 1.0
                        },
                        'strong_win_rate': 0.7,
                        'min_valid_win_rate': 0.3
                    },
                    'league_exploiter': {
                        'one_phase_step': 200,
                        'branch_probs': {
                            'pfsp': 1.0
                        },
                        'strong_win_rate': 0.7,
                        'mutate_prob': 0.5
                    },
                    'use_pretrain': False,
                    'use_pretrain_init_historical': True,
                    'pretrain_checkpoint_path': {
                        'default': 'sl_model.pth', 
                    },
                    'payoff': {
                        'type': 'battle',
                        'decay': 0.99,
                        'min_win_rate_games': 8
                    },
                    'metric': {
                        'mu': 0,
                        'sigma': 8.333333333333334,
                        'beta': 4.166666666666667,
                        'tau': 0.0,
                        'draw_probability': 0.02
                    }
                }
            },
            'type': 'ppo',
            'cuda': False,
            'on_policy': True,
            'priority': False,
            'priority_IS_weight': False,
            'recompute_adv': True,
            'action_space': 'discrete',
            'nstep_return': False,
            'multi_agent': False,
            'transition_with_policy_data': True,
            'cfg_type': 'PPOPolicyDict'
        },
        'exp_name': 'league_demo',
        'seed': 0
    }
)
