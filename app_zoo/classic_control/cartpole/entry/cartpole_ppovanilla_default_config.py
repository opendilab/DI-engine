from easydict import EasyDict

cartpole_ppovanilla_default_config = dict(
    env=dict(
        env_manager_type='base',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        actor_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='ppo_vanilla',
        import_names=['nervex.policy.ppo_vanilla'],
        on_policy=False,
        model=dict(
            obs_dim=4,
            action_dim=2,
            embedding_dim=64,
        ),
        learn=dict(
            train_step=5,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            algo=dict(
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.2,
            ),
        ),
        collect=dict(
            traj_len='inf',
            unroll_len=1,
            algo=dict(
                discount_factor=0.9,
                gae_lambda=0.95,
            ),
        ),
        command=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), ),
    ),
    replay_buffer=dict(
        replay_buffer_size=1000,
    ),
    actor=dict(
        n_sample=16,
        traj_len=200,  # cartpole max episode len
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=200,
        stop_value=195,
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
cartpole_ppovanilla_default_config = EasyDict(cartpole_ppovanilla_default_config)
main_config = cartpole_ppovanilla_default_config
