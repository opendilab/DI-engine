from easydict import EasyDict

nstep = 2
unroll_len = 6
burnin_step = 2
actor_env_num = 8
evaluator_env_num = 5
cartpole_r2d2_default_config = dict(
    env=dict(
        env_manager_type='base',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
        env_type='cartpole',
        actor_env_num=actor_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    policy=dict(
        use_cuda=False,
        policy_type='r2d2',
        on_policy=False,
        model=dict(
            obs_dim=4,
            action_dim=2,
            hidden_dim_list=[128, 128, 64],
        ),
        learn=dict(
            train_step=1,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0,
            algo=dict(
                target_update_freq=200,
                discount_factor=0.99,
                burnin_step=2,
                nstep=nstep,
                use_value_rescale=True,
            ),
        ),
        collect=dict(
            traj_len=(2 * unroll_len + nstep),
            unroll_len=(2 * nstep + burnin_step),
            env_num=actor_env_num,
            algo=dict(
                burnin_step=2,
                nstep=nstep,
            ),
        ),
        eval=dict(env_num=evaluator_env_num, ),
        command=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.05,
            decay=10000,
        ), ),
    ),
    replay_buffer=dict(
        replay_buffer_size=1000,
    ),
    actor=dict(
        n_sample=32,
        traj_len=14,
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=200,
        stop_val=195,
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
cartpole_r2d2_default_config = EasyDict(cartpole_r2d2_default_config)
main_config = cartpole_r2d2_default_config
