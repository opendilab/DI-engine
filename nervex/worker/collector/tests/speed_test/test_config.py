from easydict import EasyDict

nstep = 1
test_config = dict(
    env=dict(
        manager=dict(
            type='async_subprocess',
            wait_num=7,  # 8-1
            step_wait_timeout=0.01,
        ),
        env_kwargs=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            obs_dim=8,
            action_dim=2,
            episode_step=200,
            reset_time=0.01,
            step_time=0.003,
        ),
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
        buffer_type='naive',
        replay_buffer_size=10000,
    ),
)
test_config = EasyDict(test_config)
