from easydict import EasyDict

update_freq = 50
lunarlander_sqn_default_config = dict(
    env=dict(
        # Support ['base', 'subprocess']. 'base' is pseudo parallel and 'subprocess' is subprocess parallel.
        # 'base' is used for some light env(e.g. cartpole), whose env step time is much shorter than IPC time.
        # 'subprocess' is used for more complicated env(e.g. pong and larger than pong), which is more recommended to use in practice.
        env_manager_type='subprocess',
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, ),
        # Must use the absolute path. All the following "import_names" should obey this too.
        import_names=['app_zoo.classic_control.lunarlander.envs.lunarlander_env'],
        # Env register name (refer to function "register_env").
        env_type='lunarlander',
        # Env number respectively for collector and evaluator.
        collector_env_num=6,
        evaluator_env_num=3,
    ),
    policy=dict(
        # Whether to use cuda for network.
        use_cuda=False,
        # RL policy register name (refer to function "register_policy").
        policy_type='sqn',
        import_names=['nervex.policy.sqn'],
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # Model config used for model creating. Remember to change this, especially "obs_dim" and "action_dim" according to specific env.
        model=dict(
            obs_dim=8,
            action_dim=4,
            hidden_dim_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # learn_mode config
        learn=dict(
            # How many steps to train after collector's one collection. Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            train_iteration=update_freq,
            batch_size=64,
            learning_rate_q=0.001,
            learning_rate_alpha=0.001,
            # L2 norm weight for network parameters.
            weight_decay=0.0,
            algo=dict(
                target_theta=0.005,
                alpha=0.2,
                # Reward's future discount facotr, aka. gamma.
                discount_factor=0.99,
            ),
        ),
        # collect_mode config
        collect=dict(
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
        ),
        # command_mode config
        command=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1.,
                end=0.8,
                decay=10_000,
            ),
        ),
    ),
    # You can refer to "config/buffer_manager_default_config.py" for details.
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
            max_use=100,
        ),
    ),
    collector=dict(
        # You can use either "n_sample" or "n_episode" in collector.collect.
        # Get "n_sample" samples per collect.
        n_sample=update_freq,  # training freq
        # Get "n_episode" complete episodic trajectories per collect.
        # n_episode=8,
        traj_print_freq=1000,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        # Episode number for evaluation.
        n_episode=5,
        # Evaluate every "eval_freq" training steps.
        eval_freq=500,
        # Once evaluation reward reaches "stop_value", which means the policy converges, then the whole training can end.
        stop_value=270,
    ),
    # You can refer to "config/serial.py" for details.
    learner=dict(
        load_path='',
        hook=dict(
            save_ckpt_after_iter=dict(
                name='save_ckpt_after_iter',
                type='save_ckpt',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=500, ),
            ),
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=500, ),
            ),
        ),
    ),
    commander=dict(),
)
lunarlander_sqn_default_config = EasyDict(lunarlander_sqn_default_config)
main_config = lunarlander_sqn_default_config
