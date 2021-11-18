from easydict import EasyDict
from ding.config import parallel_transform

__policy_default_config = dict(
    use_cuda=False,
    policy_type='IL',
    model=dict(),
    learn=dict(
        train_iteration=20,
        batch_size=64,
        learning_rate=0.0002,
        algo=dict(discount_factor=0.99, ),
    ),
    collect=dict(),
    command=dict(),
)

__base_learner_default_config = dict(
    load_path='',
    use_cuda=False,
    dataloader=dict(
        batch_size=64,
        chunk_size=64,
        num_workers=0,
    ),
    hook=dict(
        load_ckpt=dict(
            name='load_ckpt',
            type='load_ckpt',
            priority=20,
            position='before_run',
        ),
        log_show=dict(
            name='log_show',
            type='log_show',
            priority=20,
            position='after_iter',
            ext_args=dict(freq=50),
        ),
        save_ckpt_after_run=dict(
            name='save_ckpt_after_run',
            type='save_ckpt',
            priority=20,
            position='after_run',
        )
    ),
)

__zergling_collector_default_config = dict(
    collector_type='zergling',
    import_names=['ding.worker.collector.zergling_parallel_collector'],
    print_freq=10,
    compressor='lz4',
    policy_update_freq=3,
    env_kwargs=dict(
        import_names=['dizoo.gfootball.envs.gfootball_env'],
        env_type='gfootball',
        collector_env_num=2,
        collector_episode_num=2,
        evaluator_env_num=2,
        evaluator_episode_num=2,
        eval_stop_val=3,
        manager=dict(shared_memory=False, ),
    ),
)

__coordinator_default_config = dict(
    collector_task_timeout=30,
    learner_task_timeout=600,
    interaction=dict(
        host='auto',
        port='auto',
    ),
    commander=dict(
        parallel_commander_type='solo',
        import_names=['ding.worker.coordinator.solo_parallel_commander'],
        collector_task_space=2,
        learner_task_space=1,
        learner_cfg=__base_learner_default_config,
        collector_cfg=__zergling_collector_default_config,
        replay_buffer_cfg=dict(buffer_name=['agent'], agent=dict(
            meta_maxlen=100000,
            max_reuse=10,
        )),
        policy=__policy_default_config,
        max_iterations=int(1e9),
        eval_interval=500,
    ),
)
__coordinator_default_config = EasyDict(__coordinator_default_config)

main_config = dict(
    coordinator=__coordinator_default_config,
    learner0=dict(
        import_names=['ding.worker.learner.comm.flask_fs_learner'],
        comm_learner_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        send_policy_freq=1,
        use_distributed=False,
    ),
    collector0=dict(
        import_names=['ding.worker.collector.comm.flask_fs_collector'],
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
    collector1=dict(
        import_names=['ding.worker.collector.comm.flask_fs_collector'],
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
)
main_config = parallel_transform(main_config)
