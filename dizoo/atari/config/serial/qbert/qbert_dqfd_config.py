from easydict import EasyDict

qbert_dqn_config = dict(
    exp_name='qbert_dqfd',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=True, reset_inplace=True)
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=1e-5,
            per_train_iter_k=10,
            expert_replay_buffer_size=10000,  # justify the buffer size of the expert buffer
        ),
        collect=dict(n_sample=100, demonstration_info_path='path'
                     ),  # Users should add their own path here (path should lead to a well-trained model)
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
qbert_dqn_config = EasyDict(qbert_dqn_config)
main_config = qbert_dqn_config
qbert_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqfd'),
)
qbert_dqn_create_config = EasyDict(qbert_dqn_create_config)
create_config = qbert_dqn_create_config
