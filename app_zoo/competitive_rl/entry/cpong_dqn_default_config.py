from easydict import EasyDict
from nervex.config import parallel_transform

traj_len = 1
nstep = 1
batch_size = 64
__policy_default_config = dict(
    use_cuda=False,
    policy_type='dqn',
    import_names=['nervex.policy.dqn'],
    on_policy=False,
    model=dict(
        # obs_dim=[4, 84, 84],
        obs_dim=[3, 210, 160],
        action_dim=3,
        embedding_dim=64,
        encoder_kwargs=dict(encoder_type='conv2d'),
    ),
    learn=dict(
        batch_size=batch_size,
        learning_rate=0.0001,
        weight_decay=0.,
        algo=dict(
            target_update_freq=500,
            discount_factor=0.99,
            nstep=nstep,
        ),
    ),
    collect=dict(
        traj_len=traj_len,
        unroll_len=1,
        algo=dict(nstep=nstep),
    ),
    command=dict(eps=dict(
        type='linear',
        start=1.,
        end=0.005,
        decay=100000,
    ), ),
)

__base_learner_default_config = dict(
    load_path='',
    use_cuda=False,
    dataloader=dict(
        batch_size=batch_size,
        chunk_size=batch_size,
        num_workers=1,
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

__one_vs_one_collector_default_config = dict(
    collector_type='one_vs_one',
    import_names=['nervex.worker.collector.one_vs_one_collector'],
    print_freq=100,
    traj_len=traj_len,
    compressor='lz4',
    policy_update_freq=10,
    env_kwargs=dict(
        import_names=['app_zoo.competitive_rl.envs.competitive_rl_env'],
        env_type='competitive_rl',
        collector_env_num=16,
        collector_episode_num=4,
        evaluator_env_num=3,
        evaluator_episode_num=1,
        eval_stop_value=20,
        opponent_type="builtin",  # opponent_type is only used in evaluator
        env_id='cPongDouble-v0',
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
        parallel_commander_type='one_vs_one',
        import_names=['nervex.worker.coordinator.one_vs_one_parallel_commander'],
        collector_task_space=2,
        learner_task_space=1,
        learner_cfg=__base_learner_default_config,
        collector_cfg=__one_vs_one_collector_default_config,
        replay_buffer_cfg=dict(
            meta_maxlen=10000,
            max_use=100,
            unroll_len=1,
            min_sample_ratio=1,
            monitor=dict(log_freq=1000),
        ),
        policy=__policy_default_config,
        max_iterations=int(1e9),
        eval_interval=50,
        league=dict(
            league_type="one_vs_one",
            import_names=['nervex.league.one_vs_one_league'],
        ),
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
        repeat_num=2,
    ),
    collector0=dict(
        import_names=['nervex.worker.collector.comm.flask_fs_collector'],
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
<<<<<<< HEAD
    # actor1=dict(
    #     import_names=['nervex.worker.actor.comm.flask_fs_actor'],
    #     comm_actor_type='flask_fs',
    #     host='auto',
    #     port='auto',
    #     path_data='./data',
    #     path_policy='.',
    #     queue_maxsize=8,
    # ),
=======
    collector1=dict(
        import_names=['nervex.worker.collector.comm.flask_fs_collector'],
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
>>>>>>> master
)
main_config = parallel_transform(main_config)
