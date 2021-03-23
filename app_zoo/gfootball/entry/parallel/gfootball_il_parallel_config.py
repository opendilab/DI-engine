from easydict import EasyDict
from nervex.config import parallel_transform

__policy_default_config = dict(
    use_cuda=False,
    policy_type='IL',
    import_names=['nervex.policy.il'],
    on_policy=False,
    model=dict(),
    learn=dict(
        train_step=20,
        batch_size=64,
        learning_rate=0.0002,
        weight_decay=0.0,
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

__zergling_actor_default_config = dict(
    actor_type='zergling',
    import_names=['nervex.worker.actor.zergling_actor'],
    print_freq=10,
    traj_len=1,
    compressor='lz4',
    policy_update_freq=3,
    env_kwargs=dict(
        import_names=['app_zoo.gfootball.envs.gfootball_env'],
        env_type='gfootball',
        actor_env_num=2,
        actor_episode_num=2,
        evaluator_env_num=2,
        evaluator_episode_num=1,
        eval_stop_val=3,
    ),
)

__coordinator_default_config = dict(
    actor_task_timeout=30,
    learner_task_timeout=600,
    interaction=dict(
        host='auto',
        port='auto',
    ),
    commander=dict(
        parallel_commander_type='solo',
        import_names=['nervex.worker.coordinator.solo_parallel_commander'],
        actor_task_space=2,
        learner_task_space=1,
        learner_cfg=__base_learner_default_config,
        actor_cfg=__zergling_actor_default_config,
        replay_buffer_cfg=dict(
            buffer_name=['agent'], agent=dict(
                meta_maxlen=100000,
                max_reuse=10,
                min_sample_ratio=1,
            )
        ),
        policy=__policy_default_config,
        max_iterations=int(1e9),
        eval_interval=100,
    ),
)
__coordinator_default_config = EasyDict(__coordinator_default_config)

main_config = dict(
    coordinator=__coordinator_default_config,
    learner0=dict(
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
        comm_learner_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        send_policy_freq=1,
        use_distributed=False,
    ),
    actor0=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
    actor1=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
)
main_config = parallel_transform(main_config)
