from easydict import EasyDict

traj_len = 3000
gfootball_il_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.gfootball.envs.gfootball_env'],
        env_type='gfootball',
        actor_env_num=4,
        evaluator_env_num=4,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='IL',
        import_names=['nervex.policy.il'],
        on_policy=False,
        model=dict(),
        learn=dict(
            train_step=20,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0,
            algo=dict(discount_factor=0.99, ),
        ),
        collect=dict(),
        command=dict(),
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
        n_episode=4,
        traj_len=traj_len,
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=4,
        eval_freq=100,
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
                ext_args=dict(freq=100, ),
            ),
        ),
    ),
    commander=dict(),
)
gfootball_il_default_config = EasyDict(gfootball_il_default_config)
main_config = gfootball_il_default_config
