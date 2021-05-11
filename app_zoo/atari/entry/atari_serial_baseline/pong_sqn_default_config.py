from easydict import EasyDict

pong_sqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        collector_tor_tor_env_num=16,
        evaluator_env_num=8,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='sqn',
        import_names=['nervex.policy.sqn'],
        on_policy=False,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_dim=[4, 84, 84],
            action_dim=6,
            hidden_dim_list=[128, 128, 512],
            head_kwargs=dict(dueling=True, ),
        ),
        learn=dict(
            train_iteration=20,
            batch_size=64,
            learning_rate_q=0.0001,
            learning_rate_alpha=0.0003,
            weight_decay=0.0,
            algo=dict(
                alpha=0.2,
                target_theta=0.005,
                discount_factor=0.99,
            ),
        ),
        collect=dict(unroll_len=1, ),
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
            max_use=100,
        ),
    ),
    collector=dict(
        n_sample=100,
        collect_print_freq=5,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=5000,
        stop_value=20,
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
pong_sqn_default_config = EasyDict(pong_sqn_default_config)
main_config = pong_sqn_default_config
