from easydict import EasyDict
from .serial import base_learner_default_config

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
        traj_len=1,
        unroll_len=1,
        algo=dict(nstep=1),
    ),
)

zergling_actor_default_config = dict(
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

coordinator_default_config = dict(
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
    learner_cfg=base_learner_default_config,
    actor_cfg=zergling_actor_default_config,
    policy=policy_default_config,
    max_iterations=10,
)
coordinator_default_config = EasyDict(coordinator_default_config)

parallel_local_default_config = dict(
    coordinator=coordinator_default_config,
    learner=dict(
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
        comm_learner_type='flask_fs',
        host='0.0.0.0',
        port=coordinator_default_config.interaction.learner.learner0[2],
        path_data='.',
        path_policy='.',
        send_policy_freq=1,
    ),
    actor0=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='0.0.0.0',
        port=coordinator_default_config.interaction.actor.actor0[2],
        path_data='.',
        path_policy='.',
        queue_maxsize=8,
    ),
    actor1=dict(
        import_names=['nervex.worker.actor.comm.flask_fs_actor'],
        comm_actor_type='flask_fs',
        host='0.0.0.0',
        port=coordinator_default_config.interaction.actor.actor1[2],
        path_data='.',
        path_policy='.',
        queue_maxsize=8,
    ),
)
parallel_local_default_config = EasyDict(parallel_local_default_config)
