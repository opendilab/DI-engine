from easydict import EasyDict

qbert_dqn_config = dict(
    env=dict(
        collector_env_num=16,
        collector_episode_num=2,
        evaluator_env_num=8,
        evaluator_episode_num=1,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(
            shared_memory=False,
        ),
    ),
    policy=dict(
        cuda=False,
        priority=True,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size_list=[128, 128, 512],
            head_kwargs=dict(head_type='base', ),
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            batch_size=32,
            learning_rate=0.0001,
            learner=dict(
                learner_num=1,
                send_policy_freq=1,
            ),
        ),
        collect=dict(
            n_sample=16,
            collector=dict(
                collector_num=2,
                update_policy_second=3,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(
                replay_buffer_size=400000,
                enable_track_used_data=True,
            ),
            commander=dict(
                collector_task_space=2,
                learner_task_space=1,
                eval_interval=30,
            ),
        ),
    ),
)
qbert_dqn_config = EasyDict(qbert_dqn_config)
main_config = qbert_dqn_config

qbert_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['app_zoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_command'),
    learner=dict(type='base', import_names=['nervex.worker.learner.base_learner']),
    collector=dict(
        type='zergling',
        import_names=['nervex.worker.collector.zergling_collector'],
    ),
    commander=dict(
        type='solo',
        import_names=['nervex.worker.coordinator.solo_parallel_commander'],
    ),
    comm_learner=dict(
        type='flask_fs',
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
    ),
    comm_collector=dict(
        type='flask_fs',
        import_names=['nervex.worker.collector.comm.flask_fs_collector'],
    ),
)
qbert_dqn_create_config = EasyDict(qbert_dqn_create_config)
create_config = qbert_dqn_create_config

qbert_dqn_system_config = dict(
    coordinator=dict(
        operator_server=dict(
            system_addr='http://nervex-server.nervex-system:8080',
            api_version='/v1alpha1',
            init_replicas_request=dict(
                collectors={
                    "replicas": 2,
                },
                learners={
                    "gpus": "0",
                    "replicas": 1,
                },
            ),
            collector_target_num=2,
            learner_target_num=1,
        ),
    ),
    path_data='/nervex/qbert/data',
    path_policy='/nervex/qbert/policy',
    communication_mode='auto',
    learner_gpu_num=1,
)
qbert_dqn_system_config = EasyDict(qbert_dqn_system_config)
system_config = qbert_dqn_system_config
