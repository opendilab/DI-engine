exp_config={
    'env': {
        'n_episode': 5,
        'stop_value': 0,
        'cfg_type': 'CooperativeNavigationDict',
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': 60,
            'reset_timeout': 60,
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'context': 'fork',
            'wait_num': 2,
            'step_wait_timeout': 0.01,
            'connect_timeout': 60,
            'cfg_type': 'SyncSubprocessEnvManagerConfig',
            'type': 'subprocess'
        },
        'num_agents': 5,
        'num_landmarks': 5,
        'max_step': 100,
        'agent_num': 5,
        'collector_env_num': 4,
        'evaluator_env_num': 2,
        'env_kwargs': {
            'import_names': ['app_zoo.multiagent_particle.envs.particle_env'],
            'env_type': 'cooperative_navigation'
        }
    },
    'policy': {
        'type': 'qmix',
        'cuda': False,
        'multi_gpu': False,
        'on_policy': True,
        'priority': False,
        'model': {
            'obs_shape': 22,
            'global_obs_shape': 30,
            'action_shape': 5,
            'hidden_size_list': [128, 128, 64],
            'mixer': True,
            'agent_num': 5
        },
        'learn': {
            'update_per_collect': 100,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'target_update_theta': 0.001,
            'discount_factor': 0.99,
            'learner': {
                'load_path': '',
                'use_distributed': False,
                'dataloader': {
                    'batch_size': 2,
                    'chunk_size': 2,
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt': {
                        'name': 'load_ckpt',
                        'type': 'load_ckpt',
                        'priority': 20,
                        'position': 'before_run'
                    },
                    'log_show': {
                        'name': 'log_show',
                        'type': 'log_show',
                        'priority': 20,
                        'position': 'after_iter',
                        'ext_args': {
                            'freq': 100
                        }
                    },
                    'save_ckpt_after_run': {
                        'name': 'save_ckpt_after_run',
                        'type': 'save_ckpt',
                        'priority': 20,
                        'position': 'after_run'
                    }
                },
                'cfg_type': 'BaseLearnerConfig'
            },
            'agent_num': 5
        },
        'collect': {
            'n_episode': 6,
            'unroll_len': 16,
            'collector': {
                'collect_print_freq': 100,
                'cfg_type': 'BaseSerialCollectorConfig'
            },
            'agent_num': 5,
            'env_num': 4
        },
        'eval': {
            'evaluator': {
                'eval_freq': 50,
                'cfg_type': 'BaseSerialEvaluatorConfig'
            },
            'agent_num': 5,
            'env_num': 2
        },
        'other': {
            'eps': {
                'type': 'exp',
                'start': 1.0,
                'end': 0.05,
                'decay': 100000
            },
            'replay_buffer': {
                'replay_buffer_size': 4096,
                'replay_buffer_start_size': 0,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'min_sample_ratio': 1.0,
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'eps': 0.01,
                'monitor': {
                    'log_freq': 2000,
                    'log_path': './log/buffer/default/',
                    'natural_expire': 10,
                    'tick_expire': 10
                },
                'cfg_type': 'PrioritizedReplayBufferConfig'
            }
        },
        'cfg_type': 'QMIXPolicyConfig'
    }
}
