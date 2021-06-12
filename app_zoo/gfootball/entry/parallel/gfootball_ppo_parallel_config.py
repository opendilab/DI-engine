from easydict import EasyDict
from nervex.config import parallel_transform

gfootball_ppo_config = dict(
    env=dict(
        collector_env_num=1,
        collector_episode_num=2,
        evaluator_env_num=1,
        evaluator_episode_num=2,
        stop_value=5,
        save_replay=False,
        render=False,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            model_type='conv1d',
            import_names=['app_zoo.gfootball.model.conv1d.conv1d']
        ),
        nstep=1,
        discount_factor=0.995,
        learn=dict(
            batch_size=32,
            learning_rate=0.001,
            learner=dict(
                learner_num=1,
                send_policy_freq=1,
            ),
        ),
        collect=dict(
            n_sample=20,
            collector=dict(
                collector_num=16,
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
                enable_track_used_data=True,
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
gfootball_ppo_config = EasyDict(gfootball_ppo_config)
main_config = gfootball_ppo_config

gfootball_ppo_create_config = dict(
    env=dict(
        import_names=['app_zoo.gfootball.envs.gfootballsp_env'],
        type='gfootball_sp',
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_command',),
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
gfootball_ppo_create_config = EasyDict(gfootball_ppo_create_config)
create_config = gfootball_ppo_create_config

gfootball_ppo_system_config = dict(
    path_data='./data',
    path_policy='./policy',
    communication_mode='auto',
    learner_multi_gpu=False,
    learner_gpu_num=0,
    coordinator=dict()
)
gfootball_ppo_system_config = EasyDict(gfootball_ppo_system_config)
system_config = gfootball_ppo_system_config
