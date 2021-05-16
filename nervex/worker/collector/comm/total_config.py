exp_config = {
    'main': {
        'env': {
            'n_episode': 5,
            'stop_value': 195,
            'cfg_type': 'CartPoleEnvConfig',
            'manager': {
                'episode_num': float("inf"),
                'max_retry': 1,
                'step_timeout': 60,
                'reset_timeout': 60,
                'retry_waiting_time': 0.1,
                'shared_memory': True,
                'context': 'fork',
                'wait_num': 2,
                'step_wait_timeout': 0.01,
                'connect_timeout': 60,
                'cfg_type': 'SyncSubprocessEnvManagerConfig',
                'type': 'subprocess'
            },
            'collector_env_num': 8,
            'collector_episode_num': 1,
            'evaluator_env_num': 5,
            'evaluator_episode_num': 1,
            'type': 'cartpole',
            'import_names': ['app_zoo.classic_control.cartpole.envs.cartpole_env']
        },
        'policy': {
            'type': 'dqn_command',
            'cuda': False,
            'multi_gpu': False,
            'on_policy': False,
            'priority': False,
            'learn': {
                'update_per_collect': 3,
                'batch_size': 64,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'target_update_freq': 100,
                'discount_factor': 0.97,
                'nstep': 3,
                'ignore_done': False,
                'learner': {
                    'learner_num': 1,
                    'send_policy_second': 1
                }
            },
            'collect': {
                'n_sample': 16,
                'unroll_len': 1,
                'nstep': 3,
                'her': False,
                'her_strategy': 'future',
                'her_replay_k': 1,
                'collector': {
                    'print_freq': 5,
                    'compressor': 'lz4',
                    'update_policy_second': 3,
                    'cfg_type': 'ZerglingCollectorDict',
                    'collector_num': 2,
                    'type': 'zergling',
                    'import_names': ['nervex.worker.collector.zergling_collector']
                }
            },
            'eval': {
                'evaluator': {
                    'eval_freq': 50
                }
            },
            'other': {
                'eps': {
                    'type': 'exp',
                    'start': 0.95,
                    'end': 0.1,
                    'decay': 10000
                },
                'replay_buffer': {
                    'buffer_type': 'priority',
                    'replay_buffer_size': 100000,
                    'replay_buffer_start_size': 0,
                    'max_use': float("inf"),
                    'max_staleness': float("inf"),
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
                },
                'commander': {
                    'collector_task_space': 2,
                    'learner_task_space': 1,
                    'eval_interval': 10,
                    'cfg_type': 'NaiveCommanderDict',
                    'type': 'naive',
                    'import_names': ['nervex.worker.coordinator.base_parallel_commander']
                }
            },
            'cfg_type': 'DQNCommandModePolicyConfig',
            'model': {
                'obs_shape': 4,
                'action_shape': 2,
                'hidden_size_list': [128, 128, 64],
                'dueling': True
            }
        }
    },
    'system': {
        'coordinator': {
            'collector_task_timeout': 30,
            'learner_task_timeout': 600,
            'cfg_type': 'CoordinatorDict',
            'host': '0.0.0.0',
            'port': 64614,
            'learner': {
                'learner0': ['learner0', '0.0.0.0', 64615]
            },
            'collector': {
                'collector0': ['collector0', '0.0.0.0', 64616],
                'collector1': ['collector1', '0.0.0.0', 64617]
            }
        },
        'learner0': {
            'type': 'flask_fs',
            'import_names': ['nervex.worker.learner.comm.flask_fs_learner'],
            'host': '0.0.0.0',
            'port': 64615,
            'path_data': '.',
            'path_policy': '.'
        },
        'collector0': {
            'type': 'flask_fs',
            'import_names': ['nervex.worker.collector.comm.flask_fs_collector'],
            'host': '0.0.0.0',
            'port': 64616,
            'path_data': '.',
            'path_policy': '.'
        },
        'collector1': {
            'type': 'flask_fs',
            'import_names': ['nervex.worker.collector.comm.flask_fs_collector'],
            'host': '0.0.0.0',
            'port': 64617,
            'path_data': '.',
            'path_policy': '.'
        }
    }
}
