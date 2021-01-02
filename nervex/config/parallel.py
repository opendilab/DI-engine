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
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.,
        algo=dict(
            target_update_freq=100,
            discount_factor=0.95,
            nstep=1,
        ),
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
    policy=policy_default_config,
)
coordinator_default_config = EasyDict(coordinator_default_config)

parallel_local_default_config = dict(
    coordinator=coordinator_default_config,
    learner=dict(
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
        comm_learner_type='flask_fs',
        host=coordinator_default_config.interaction.learner.learner0[1],
        port=coordinator_default_config.interaction.learner.learner0[2],
        path_traj='.',
        path_agent='.',
        send_agent_freq=1,
    ),
)
parallel_local_default_config = EasyDict(parallel_local_default_config)
