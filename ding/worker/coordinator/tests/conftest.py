import pytest
from easydict import EasyDict

from ding.config import compile_config_parallel
from ding.worker.coordinator.one_vs_one_parallel_commander import OneVsOneCommander


@pytest.fixture(scope='function')
def setup_1v1commander():
    nstep = 1
    eval_interval = 5
    main_config = dict(
        exp_name='one_vs_one_test',
        env=dict(
            collector_env_num=8,
            collector_episode_num=2,
            evaluator_env_num=5,
            evaluator_episode_num=1,
            stop_value=20,
        ),
        policy=dict(
            cuda=False,
            model=dict(
                obs_shape=[4, 84, 84],
                action_shape=3,
                encoder_kwargs=dict(encoder_type='conv2d'),
            ),
            nstep=nstep,
            learn=dict(
                batch_size=32,
                learning_rate=0.0001,
                weight_decay=0.,
                algo=dict(
                    target_update_freq=500,
                    discount_factor=0.99,
                    nstep=nstep,
                ),
                learner=dict(
                    learner_num=1,
                    send_policy_freq=1,
                ),
            ),
            collect=dict(
                traj_len=15,
                algo=dict(nstep=nstep),
                collector=dict(
                    collector_num=2,
                    update_policy_second=3,
                ),
            ),
            other=dict(
                eps=dict(
                    type='linear',
                    start=1.,
                    end=0.005,
                    decay=1000000,
                ),
                commander=dict(
                    collector_task_space=2,
                    learner_task_space=1,
                    eval_interval=eval_interval,
                    league=dict(naive_sp_player=dict(one_phase_step=1000, ), ),
                ),
                replay_buffer=dict(),
            ),
        ),
    )
    main_config = EasyDict(main_config)
    create_config = dict(
        env=dict(
            # 1v1 commander should use “competitive_rl”.
            # However, because this env is hard to install, we use "cartpole" instead.
            # But commander does not need a real env, it is just preserved to use `compile_config_parallel`.
            type='cartpole',
            import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn_command'),
        learner=dict(type='base', import_names=['ding.worker.learner.base_learner']),
        collector=dict(
            type='zergling',
            import_names=['ding.worker.collector.zergling_parallel_collector'],
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
        league=dict(type='one_vs_one'),
    )
    system_config = dict(
        coordinator=dict(),
        path_data='./data',
        path_policy='./policy',
        communication_mode='auto',
        learner_gpu_num=1,
    )
    system_config = EasyDict(system_config)
    create_config = EasyDict(create_config)
    config = compile_config_parallel(main_config, create_cfg=create_config, system_cfg=system_config)
    return OneVsOneCommander(config['main'])
