from easydict import EasyDict

bitflip_dqn_default_config = dict(
    env=dict(
        env_manager_type='base',
        import_names=['app_zoo.classic_control.bitflip.envs.bitflip_env'],
        env_type='bitflip',
        actor_env_num=1,
        evaluator_env_num=8,
        n_bits=5,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='dqn',
        import_names=['nervex.policy.dqn'],
        on_policy=False,
        model=dict(
            obs_dim=10,
            action_dim=5,
            embedding_dim=64,
            dueling=False,
        ),
        learn=dict(
            train_step=1,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0,
            algo=dict(
                nstep=1,
                target_update_freq=500,
                discount_factor=0.9,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(
                nstep=1,
                use_her=True,
                her_strategy='final',
                her_replay_k=1,
            ),
        ),
        command=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.02,
            decay=10000,
        ), ),
    ),
    replay_buffer=dict(
        replay_buffer_size=5000,
    ),
    actor=dict(
        n_episode=1,
        traj_len=50,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=16,
        eval_freq=100,
        stop_val=0.9,
    ),
    learner=dict(
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
bitflip_dqn_default_config = EasyDict(bitflip_dqn_default_config)
main_config = bitflip_dqn_default_config
