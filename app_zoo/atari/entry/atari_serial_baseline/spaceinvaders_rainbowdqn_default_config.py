from easydict import EasyDict

nstep = 5
spaceinvaders_rainbowdqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='SpaceInvadersNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        collector_env_num=16,
        evaluator_env_num=4,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='rainbow_dqn',
        import_names=['nervex.policy.rainbow_dqn'],
        on_policy=False,
        use_priority=True,
        model=dict(
            encoder_kwargs=dict(encoder_type='conv2d', ),
            obs_dim=[4, 84, 84],
            action_dim=6,
            hidden_dim_list=[128, 128, 256],
            head_kwargs=dict(
                dueling=True,
                distribution=True,
                noise=True,
            ),
            v_max=10,
            v_min=-10,
            n_atom=51,
        ),
        learn=dict(
            train_iteration=20,
            batch_size=32,
            learning_rate=0.0001,
            algo=dict(
                target_update_freq=400,
                discount_factor=0.99,
                nstep=nstep,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        command=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.05,
            decay=50000,
        ), ),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            replay_buffer_size=100000,
            max_use=100,
        ),
    ),
    collector=dict(
        n_sample=400,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=500,
        stop_value=700,
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
spaceinvaders_rainbowdqn_default_config = EasyDict(spaceinvaders_rainbowdqn_default_config)
main_config = spaceinvaders_rainbowdqn_default_config
