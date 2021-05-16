from easydict import EasyDict

nstep = 3
cartpole_dqn_config = dict(
    env=dict(
        collector_env_num=8,
        collector_episode_num=2,
        evaluator_env_num=5,
        evaluator_episode_num=1,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            discount_factor=0.97,
            nstep=nstep,
            learner=dict(
                learner_num=1,
                send_policy_freq=1,
            ),
        ),
        collect=dict(
            n_sample=16,
            nstep=nstep,
            collector=dict(
                collector_num=2,
                update_policy_second=3,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
                enable_track_used_data=False,
            ),
            commander=dict(
                collector_task_space=2,
                learner_task_space=1,
                eval_interval=5,
            ),
        ),
    ),
)
cartpole_dqn_config = EasyDict(cartpole_dqn_config)
main_config = cartpole_dqn_config

cartpole_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_command'),
    learner=dict(
        type='base',
        import_names=['nervex.worker.learner.base_learner']
    ),
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
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config

cartpole_dqn_system_config = dict(
    path_data='./data',
    path_policy='./policy',
    communication_mode='auto',
    learner_multi_gpu=False,
)
cartpole_dqn_system_config = EasyDict(cartpole_dqn_system_config)
system_config = cartpole_dqn_system_config
