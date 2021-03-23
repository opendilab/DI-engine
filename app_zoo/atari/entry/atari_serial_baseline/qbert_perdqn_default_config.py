from easydict import EasyDict

traj_len = 20
nstep = 3
qbert_dqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        actor_env_num=16,
        evaluator_env_num=4,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='dqn',
        import_names=['nervex.policy.dqn'],
        on_policy=False,
        use_priority=True,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_dim=[4, 84, 84],
            action_dim=6,
            hidden_dim_list=[128, 128, 512],
            head_kwargs=dict(dueling=True, ),
        ),
        learn=dict(
            train_step=10,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.00001,
            algo=dict(
                target_update_freq=500,
                discount_factor=0.99,
                nstep=nstep,
            ),
        ),
        collect=dict(
            traj_len=traj_len,
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        command=dict(eps=dict(
            type='linear',
            start=1.,
            end=0.05,
            decay=100000,
        ), ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
            max_reuse=1000,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=100,
        traj_len=traj_len,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=4,
        eval_freq=2000,
        stop_val=9000,
    ),
    learner=dict(
        load_path='',
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=200, ),
            ),
        ),
    ),
    commander=dict(),
)
qbert_dqn_default_config = EasyDict(qbert_dqn_default_config)
main_config = qbert_dqn_default_config
