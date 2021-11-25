from easydict import EasyDict

cartpole_dqn_config = dict(
    exp_name='cartpole_dqn',
    env=dict(
        collector_env_num=8,
        collector_episode_num=2,
        evaluator_env_num=5,
        evaluator_episode_num=1,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.97,
        learn=dict(
            batch_size=32,
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
                # increase collector task space when get rs from server
                collector_task_space=0,
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
        type='solo',
        import_names=['ding.worker.coordinator.solo_parallel_commander'],
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
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config

cartpole_dqn_system_config = dict(
    coordinator=dict(
        operator_server=dict(
            system_addr='di-server.di-system:8080',
            api_version='/v1alpha1',
            init_replicas_request=dict(
                collectors={
                    "replicas": 2,
                },
                learners={
                    "gpus": "0",
                    "replicas": 1,
                },
            ),
            collector_target_num=2,
            learner_target_num=1,
        ),
    ),
    path_data='./{}/data'.format(main_config.exp_name),
    path_policy='./{}/policy'.format(main_config.exp_name),
    communication_mode='auto',
    learner_gpu_num=1,
)
cartpole_dqn_system_config = EasyDict(cartpole_dqn_system_config)
system_config = cartpole_dqn_system_config
