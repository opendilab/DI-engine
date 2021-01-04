from easydict import EasyDict

__policy_default_config = dict(
    use_cuda=False,
    policy_type='dqn',
    import_names=['nervex.policy.dqn'],
    on_policy=False,
    model=dict(
        obs_dim=4,
        action_dim=2,
    ),
    learn=dict(
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.,
        algo=dict(
            target_update_freq=100,
            discount_factor=0.95,
            nstep=1,
        ),
    ),
    collect=dict(
        traj_len=1,
        unroll_len=1,
        algo=dict(nstep=1),
    ),
)

__base_learner_default_config = dict(
    save_path='.',
    load_path='',
    use_cuda=False,
    use_distributed=False,
    dataloader=dict(
        batch_size=2,
        chunk_size=2,
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
            ext_args=dict(freq=1),
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
    save_path='.',
    actor_type='zergling',
    import_names=['nervex.worker.actor.zergling_actor'],
    print_freq=10,
    traj_len=1,
    compressor='lz4',
    policy_update_freq=3,
    policy_update_path='test.pth',
    env_kwargs=dict(
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        actor_env_num=8,
        evaluator_env_num=5,
        episode_num=2,
    ),
)

__coordinator_default_config = dict(
    actor_task_timeout=30,
    learner_task_timeout=600,
    interaction=dict(
        host='0.0.0.0',
        port=12345,
        learner=dict(learner0=['learner0', '0.0.0.0', 11110]),
        actor=dict(
            actor0=['actor0', '0.0.0.0', 11111],
            actor1=['actor1', '0.0.0.0', 11112],
        ),
    ),
    commander=dict(
        parallel_commander_type='solo',
        import_names=['nervex.worker.coordinator.solo_parallel_commander'],
        actor_task_space=2,
        learner_task_space=1,
        learner_cfg=__base_learner_default_config,
        actor_cfg=__zergling_actor_default_config,
        policy=__policy_default_config,
        max_iterations=int(1e9),
    ),
)
__coordinator_default_config = EasyDict(__coordinator_default_config)

main_config = dict(
    coordinator=__coordinator_default_config,
    learner0=dict(
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
        comm_learner_type='flask_fs',
        host='0.0.0.0',
        port=__coordinator_default_config.interaction.learner.learner0[2],
        path_data='./data',
        path_policy='.',
        send_policy_freq=1,
    ),
    actor0=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='0.0.0.0',
        port=__coordinator_default_config.interaction.actor.actor0[2],
        path_data='.',
        path_policy='.',
        queue_maxsize=8,
    ),
    actor1=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='0.0.0.0',
        port=__coordinator_default_config.interaction.actor.actor1[2],
        path_data='./data',
        path_policy='.',
        queue_maxsize=8,
    ),
)
main_config = EasyDict(main_config)
