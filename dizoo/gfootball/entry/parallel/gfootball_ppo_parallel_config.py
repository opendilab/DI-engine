from easydict import EasyDict
from ding.config import parallel_transform
from copy import deepcopy

gfootball_ppo_config = dict(
    env=dict(
        collector_env_num=1,
        collector_episode_num=1,
        evaluator_env_num=1,
        evaluator_episode_num=1,
        stop_value=5,
        save_replay=False,
        render=False,
    ),
    policy=dict(
        cuda=False,
        model=dict(type='conv1d', import_names=['dizoo.gfootball.model.conv1d.conv1d']),
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
            env_num=1,
            collector=dict(
                collector_num=1,
                update_policy_second=3,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=50), env_num=1),
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
        import_names=['dizoo.gfootball.envs.gfootballsp_env'],
        type='gfootball_sp',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_lstm_command', import_names=['dizoo.gfootball.policy.ppo_lstm']),
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
gfootball_ppo_create_config = EasyDict(gfootball_ppo_create_config)
create_config = gfootball_ppo_create_config

gfootball_ppo_system_config = dict(
    path_data='./data',
    path_policy='./policy',
    communication_mode='auto',
    learner_multi_gpu=False,
    learner_gpu_num=1,
    coordinator=dict()
)
gfootball_ppo_system_config = EasyDict(gfootball_ppo_system_config)
system_config = gfootball_ppo_system_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c gfootball_ppo_parallel_config.py -s 0`
    from ding.entry import parallel_pipeline
    config = tuple([deepcopy(main_config), deepcopy(create_config), deepcopy(system_config)])
    parallel_pipeline(config, seed=0)
