Parallel Pipeline Config
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    traj_len=1

    # You can refer to policy config in serial pipeline config for details
    policy_default_config = dict(
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
            traj_len=traj_len,
            unroll_len=1,
            algo=dict(
                nstep=1,
            ),
        ),
    )

    zergling_actor_default_config = dict(
        actor_type='zergling',
        import_names=['nervex.worker.actor.zergling_actor'],
        print_freq=10,
        traj_len=traj_len,
        # The function to compress data.
        compressor='lz4',
        # Frequency for actor to update its own policy according learner's saved one.
        policy_update_freq=3,
        policy_update_path='test.pth',
        # Env config for actor and evaluator.
        env_kwargs=dict(
            env_type='cartpole',
            actor_env_num=8,
            evaluator_env_num=5,
            actor_episode_num=2,
            evaluator_episode_num=1,
            eval_stop_val=1e9,
        ),
    )

    coordinator_default_config = dict(
        # Timeout seconds for actor and learner.
        actor_task_timeout=30,
        learner_task_timeout=600,
        # For host and port settings, can be 'auto' (allocate according to current situation) or specific one.
        # Host and port in learner and actor config are the same.
        interaction=dict(
            host='auto',
            port='auto',
        ),
        commander=dict(
            parallel_commander_type='naive',
            import_names=[],
            # Task space for actor and learner.
            actor_task_space=2,
            learner_task_space=1,
            learner_cfg=base_learner_default_config,
            actor_cfg=zergling_actor_default_config,
            replay_buffer_cfg=dict(),
            policy=policy_default_config,
            max_iterations=10,
        ),
    )
    coordinator_default_config = EasyDict(coordinator_default_config)

    parallel_local_default_config = dict(
        coordinator=coordinator_default_config,
        # In general, learner number and actor should be in accordance with commander's task space.
        # Here we have 1 learner and 2 actors.
        learner0=dict(
            import_names=['nervex.worker.learner.comm.flask_fs_learner'],
            comm_learner_type='flask_fs',
            host='auto',
            port='auto',
            # Path for loading data.
            path_data='.',
            # Path for saving policy.
            path_policy='.',
            # Frequence for saving learner's policy to file.
            send_policy_freq=1,
            repeat_num=1,
            # Whether to used cross-machine distributed training.
            use_distributed=False,
        ),
        actor0=dict(
            import_names=['nervex.worker.actor.comm.flask_fs_actor'],
            comm_actor_type='flask_fs',
            host='auto',
            port='auto',
            path_data='.',
            path_policy='.',
            queue_maxsize=8,
        ),
        actor1=dict(
            import_names=['nervex.worker.actor.comm.flask_fs_actor'],
            comm_actor_type='flask_fs',
            host='auto',
            port='auto',
            path_data='.',
            path_policy='.',
            queue_maxsize=8,
        ),
    )
