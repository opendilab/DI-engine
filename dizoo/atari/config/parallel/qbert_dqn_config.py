from easydict import EasyDict

qbert_dqn_config = dict(
    exp_name='qbert_dqn',
    env=dict(
        collector_env_num=16,
        collector_episode_num=2,
        evaluator_env_num=8,
        evaluator_episode_num=1,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        #'ALE/Qbert-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
        manager=dict(shared_memory=True, ),
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
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
            n_sample=32,
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
                eval_interval=300,
            ),
        ),
    ),
)
qbert_dqn_config = EasyDict(qbert_dqn_config)
main_config = qbert_dqn_config

qbert_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_command'),
    learner=dict(type='base', import_names=['ding.worker.learner.base_learner']),
    collector=dict(
        type='zergling',
        import_names=['ding.worker.collector.zergling_parallel_collector'],
    ),
    commander=dict(
        type='solo',
        import_names=['ding.worker.coordinator.solo_parallel_commander'],
    ),
    comm_learner=dict(
        type='flask_fs',
        import_names=['ding.worker.learner.comm.flask_fs_learner'],
    ),
    comm_collector=dict(
        type='flask_fs',
        import_names=['ding.worker.collector.comm.flask_fs_collector'],
    ),
)
qbert_dqn_create_config = EasyDict(qbert_dqn_create_config)
create_config = qbert_dqn_create_config

qbert_dqn_system_config = dict(
    coordinator=dict(),
    path_data='./{}/data'.format(main_config.exp_name),
    path_policy='./{}/policy'.format(main_config.exp_name),
    communication_mode='auto',
    learner_gpu_num=1,
)
qbert_dqn_system_config = EasyDict(qbert_dqn_system_config)
system_config = qbert_dqn_system_config
