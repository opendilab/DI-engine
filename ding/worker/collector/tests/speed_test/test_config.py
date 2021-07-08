from easydict import EasyDict

nstep = 1
test_config = dict(
    env=dict(
        manager=dict(
            type='async_subprocess',
            wait_num=7,  # 8-1
            step_wait_timeout=0.01,
        ),
        collector_env_num=8,
        evaluator_env_num=5,
        obs_dim=8,
        action_dim=2,
        episode_step=200,
        reset_time=0.01,
        step_time=0.003,
    ),
    policy=dict(
        use_cuda=False,
        forward_time=0.002,
        learn=dict(),
        collect=dict(
            n_sample=80,
            unroll_len=1,
            collector=dict(),
        ),
        eval=dict(),
        other=dict(replay_buffer=dict(
            type='naive',
            replay_buffer_size=10000,
        ), ),
    ),
)
test_config = EasyDict(test_config)
