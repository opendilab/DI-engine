from easydict import EasyDict

lunarlander_dqfd_config = dict(
    exp_name='lunarlander_dqfd',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=1e-5,
            per_train_iter_k=10,
            expert_replay_buffer_size=10000,  # justify the buffer size of the expert buffer
        ),
        collect=dict(
            n_sample=64,
            # Users should add their own path here (path should lead to a well-trained model)
            demonstration_info_path='path',
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),  # note: this is the times after which you learns to evaluate
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
lunarlander_dqfd_config = EasyDict(lunarlander_dqfd_config)
main_config = lunarlander_dqfd_config
lunarlander_dqfd_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqfd'),
)
lunarlander_dqfd_create_config = EasyDict(lunarlander_dqfd_create_config)
create_config = lunarlander_dqfd_create_config
