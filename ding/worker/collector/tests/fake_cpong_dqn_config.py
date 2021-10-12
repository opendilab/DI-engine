from easydict import EasyDict
from ding.config import parallel_transform

fake_cpong_dqn_config = dict(
    exp_name='fake_cpong_dqn',
    env=dict(
        collector_env_num=16,
        collector_episode_num=2,
        evaluator_env_num=8,
        evaluator_episode_num=2,
        stop_value=20,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 256],
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
                update_policy_second=5,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=5, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
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
fake_cpong_dqn_config = EasyDict(fake_cpong_dqn_config)
main_config = fake_cpong_dqn_config

fake_cpong_dqn_create_config = dict(
    env=dict(
        import_names=['ding.worker.collector.tests.test_marine_parallel_collector'],
        type='fake_competitive_rl',
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_command'),
    learner=dict(type='base', import_names=['ding.worker.learner.base_learner']),
    collector=dict(
        type='marine',
        import_names=['ding.worker.collector.marine_parallel_collector'],
    ),
    commander=dict(
        type='one_vs_one',
        import_names=['ding.worker.coordinator.one_vs_one_parallel_commander'],
    ),
    comm_learner=dict(
        type='flask_fs',
        import_names=['ding.worker.learner.comm.flask_fs_learner'],
    ),
    comm_collector=dict(
        type='flask_fs',
        import_names=['ding.worker.collector.comm.flask_fs_collector'],
    ),
)
fake_cpong_dqn_create_config = EasyDict(fake_cpong_dqn_create_config)
create_config = fake_cpong_dqn_create_config

fake_cpong_dqn_system_config = dict(
    coordinator=dict(),
    path_data='./data',
    path_policy='./policy',
    communication_mode='auto',
    learner_gpu_num=0,
)
fake_cpong_dqn_system_config = EasyDict(fake_cpong_dqn_system_config)
system_config = fake_cpong_dqn_system_config
