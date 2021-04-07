from easydict import EasyDict

nstep = 1
test_config = dict(
    env=dict(
        env_manager_type='async_subprocess',
        manager=dict(shared_memory=True, ),
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        actor_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        use_cuda=False,
    ),
    actor=dict(
        # You can use either "n_sample" or "n_episode" in actor.collect.
        # Get "n_sample" samples per collect.
        n_sample=8,
        # Get "n_episode" complete episodic trajectories per collect.
        # n_episode=8,
        traj_len=1,
        collect_print_freq=100,
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=300,
            max_use=100,
            min_sample_ratio=1,
        ),
    ),
)
test_config = EasyDict(test_config)
