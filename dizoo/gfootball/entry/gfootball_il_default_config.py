from easydict import EasyDict

gfootball_il_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
        env_type='gfootball',
        collector_env_num=4,
        evaluator_env_num=2,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        use_cuda=True,
        policy_type='IL',
        model=dict(),
        learn=dict(
            train_iteration=20,
            batch_size=64,
            learning_rate=0.0002,
            algo=dict(discount_factor=0.99, ),
        ),
        collect=dict(),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
            max_reuse=10,
        ),
    ),
    collector=dict(
        n_episode=2,
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=4,
        eval_freq=800,
        stop_val=3,
    ),
    learner=dict(
        load_path='',
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=20, ),
            ),
        ),
    ),
    commander=dict(),
)
gfootball_il_default_config = EasyDict(gfootball_il_default_config)
main_config = gfootball_il_default_config
