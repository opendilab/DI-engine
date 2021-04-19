from easydict import EasyDict

from .serial import base_learner_default_config

traj_len = 1

# You can refer to policy config in serial pipeline config for details
policy_default_config = dict(
    use_cuda=False,
    policy_type='dqn',
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
        algo=dict(nstep=1, ),
    ),
)

zergling_collector_default_config = dict(
    collector_type='zergling',
    print_freq=10,
    traj_len=traj_len,
    # The function to compress data.
    compressor='lz4',
    # Frequency for collector to update its own policy according learner's saved one.
    policy_update_freq=3,
    policy_update_path='test.pth',
    # Env config for collector and evaluator.
    env_kwargs=dict(
        env_type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        collector_env_num=8,
        evaluator_env_num=5,
        collector_episode_num=2,
        evaluator_episode_num=1,
        eval_stop_value=1e9,
    ),
)

coordinator_default_config = dict(
    # Timeout seconds for collector and learner.
    collector_task_timeout=30,
    learner_task_timeout=600,
    # For host and port settings, can be 'auto' (allocate according to current situation) or specific one.
    # Host and port in learner and collector config are the same.
    interaction=dict(
        host='auto',
        port='auto',
    ),
    commander=dict(
        parallel_commander_type='naive',
        # Task space for collector and learner.
        collector_task_space=2,
        learner_task_space=1,
        learner_cfg=base_learner_default_config,
        collector_cfg=zergling_collector_default_config,
        replay_buffer_cfg=dict(),
        policy=policy_default_config,
        max_iterations=10,
    ),
)
coordinator_default_config = EasyDict(coordinator_default_config)

parallel_local_default_config = dict(
    coordinator=coordinator_default_config,
    # In general, learner number and collector number should be in accordance with commander's task space.
    # Here we have 1 learner and 2 collectors.
    learner0=dict(
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
    collector0=dict(
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='.',
        path_policy='.',
        queue_maxsize=8,
    ),
    collector1=dict(
        comm_collector_type='flask_fs',
        host='auto',
        port='auto',
        path_data='.',
        path_policy='.',
        queue_maxsize=8,
    ),
)
