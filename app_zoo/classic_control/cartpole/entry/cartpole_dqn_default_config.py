from easydict import EasyDict

nstep = 1
cartpole_dqn_default_config = dict(
    env=dict(
        manager=dict(
            # Support ['base', 'subprocess']. 'base' is pseudo parallel and 'subprocess' is subprocess parallel.
            # 'base' is used for some light env(e.g. cartpole), whose env step time is much shorter than IPC time.
            # 'subprocess' is used for more complicated env(e.g. pong and larger than pong), which is more recommended to use in practice.
            type='base',
        ),
        env_kwargs=dict(
            # Must use the absolute path. All the following "import_names" should obey this too.
            import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
            # Env register name (refer to function "register_env").
            env_type='cartpole',
            # Env number respectively for actor and evaluator.
            actor_env_num=8,
            evaluator_env_num=5,
        )
    ),
    policy=dict(
        # Whether to use cuda for network.
        use_cuda=False,
        # RL policy register name (refer to function "register_policy").
        policy_type='dqn',
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # Model config used for model creating. Remember to change this, especially "obs_dim" and "action_dim" according to specific env.
        model=dict(
            obs_dim=4,
            action_dim=2,
            hidden_dim_list=[128, 128, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # learn_mode config
        learn=dict(
            # How many steps to train after actor's one collection. Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            train_iteration=3,
            batch_size=64,
            learning_rate=0.001,
            # L2 norm weight for network parameters.
            weight_decay=0.0,
            algo=dict(
                # Frequence of target network update.
                target_update_freq=100,
                # Reward's future discount facotr, aka. gamma.
                discount_factor=0.97,
                # How many steps in td error.
                nstep=nstep,
            ),
        ),
        # collect_mode config
        collect=dict(
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
        ),
    ),
    # You can refer to "config/buffer_manager_default_config.py" for details.
    replay_buffer=dict(
        replay_buffer_size=20000,
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
    evaluator=dict(
        # Episode number for evaluation.
        n_episode=5,
        # Evaluate every "eval_freq" training steps.
        eval_freq=50,
        # Once evaluation reward reaches "stop_value", which means the policy converges, then the whole training can end.
        stop_value=195,
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
                ext_args=dict(freq=100, ),
            ),
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=100, ),
            ),
        ),
    ),
)
cartpole_dqn_default_config = EasyDict(cartpole_dqn_default_config)
main_config = cartpole_dqn_default_config
