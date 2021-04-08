from easydict import EasyDict

nstep = 1
test_config = dict(
    env=dict(
        env_manager_type='subprocess',
        manager=dict(shared_memory=True, ),
        collector_env_num=8,
        evaluator_env_num=5,
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        obs_dim=8,
        action_dim=2,
        episode_step=200,
        reset_time=0.01,
        step_time=0.003,
    ),
    policy=dict(
        use_cuda=False,
        forward_time=0.002,
    ),
    collector=dict(
        n_sample=80,
        # n_episode=8,
        traj_len=1,
        collect_print_freq=5000,
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=300,
            max_use=100,
            min_sample_ratio=1,
            monitor=dict(
                log_freq=5000,
                natural_expire=3,
                tick_expire=3,
            ),
        ),
    ),
)
test_config = EasyDict(test_config)
