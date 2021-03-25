from easydict import EasyDict

traj_len = 11
nstep = 3
miner_dqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.procgen.miner.envs.miner_env'],
        env_type='miner',
        # frame_stack=4,
        is_train=True,
        actor_env_num=16,
        evaluator_env_num=8,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='dqn',
        import_names=['nervex.policy.dqn'],
        on_policy=False,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_dim=[3, 64, 64],
            action_dim=1,
            embedding_dim=512,
            head_kwargs=dict(dueling=False, ),
        ),
        learn=dict(
            train_step=20,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.0,
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
            type='exp',
            start=1.,
            end=0.05,
            decay=200000,
        ), ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=10000,
            max_reuse=100,
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
        eval_freq=5000,
        stop_val=20,
    ),
    learner=dict(
        load_path='',
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=100, ),
            ),
        ),
    ),
    commander=dict(),
)
miner_dqn_default_config = EasyDict(miner_dqn_default_config)
main_config = miner_dqn_default_config
