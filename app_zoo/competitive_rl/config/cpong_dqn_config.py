from easydict import EasyDict
from nervex.config import parallel_transform

cpong_dqn_config = dict(
    env=dict(
        collector_env_num=16,
        collector_episode_num=2,
        evaluator_env_num=8,
        evaluator_episode_num=2,
        stop_value=20,
        opponent_type="builtin",  # opponent_type is only used in evaluator
        env_id='cPongDouble-v0',
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            embedding_dim=64,
            encoder_kwargs=dict(encoder_type='conv2d'),
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            batch_size=16,
            learning_rate=0.001,
            learner=dict(
                learner_num=1,
                send_policy_freq=1,
            ),
        ),
        collect=dict(
            n_sample=16,
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
                league=dict(),
            ),
        ),
    )
)
cpong_dqn_config = EasyDict(cpong_dqn_config)
main_config = cpong_dqn_config

cpong_dqn_create_config = dict(
    env=dict(
        import_names=['app_zoo.competitive_rl.envs.competitive_rl_env'],
        type='competitive_rl',
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_command'),
    learner=dict(type='base', import_names=['nervex.worker.learner.base_learner']),
    collector=dict(
        type='one_vs_one',
        import_names=['nervex.worker.collector.one_vs_one_collector'],
    ),
    commander=dict(
        type='one_vs_one',
        import_names=['nervex.worker.coordinator.one_vs_one_parallel_commander'],
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
cpong_dqn_create_config = EasyDict(cpong_dqn_create_config)
create_config = cpong_dqn_create_config

cpong_dqn_system_config = dict(
    path_data='./data',
    path_policy='./policy',
    communication_mode='auto',
    learner_multi_gpu=False,
)
cpong_dqn_system_config = EasyDict(cpong_dqn_system_config)
system_config = cpong_dqn_system_config
