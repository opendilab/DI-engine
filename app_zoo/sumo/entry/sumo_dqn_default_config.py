from easydict import EasyDict

nstep = 1
traj_len = 1
sumo_dqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, ),
        # Must use the absolute path. All the following "import_names" should obey this too.
        import_names=['app_zoo.sumo.envs.sumo_env'],
        env_type='sumo_wj3',
        actor_env_num=4,
        evaluator_env_num=1,
    ),
    policy=dict(
        # Whether to use cuda for network.
        use_cuda=False,
        # RL policy register name (refer to function "register_policy").
        policy_type='md_dqn',
        import_names=['app_zoo.common.policy.md_dqn'],
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # Model config used for model creating. Remember to change this, especially "obs_dim" and "action_dim" according to specific env.
        model=dict(
            obs_dim=380,
            action_dim=[2, 2, 3],
            embedding_dim=64,
            # Whether to use dueling head.
            dueling=True,
        ),
        # learn_mode config
        learn=dict(
            # How many steps to train after actor's one collection. Bigger "train_step" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            train_step=3,
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
            # Will collect trajectory with length "traj_len".
            traj_len=traj_len,
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        # command_mode config
        command=dict(
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
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=100000,
            max_use=100,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        # You can use either "n_sample" or "n_episode" in actor.collect.
        # Get "n_sample" samples per collect.
        n_sample=8,
        # Get "n_episode" complete episodic trajectories per collect.
        # n_episode=8,
        traj_len=traj_len,
        collect_print_freq=100,
    ),
    evaluator=dict(
        # Episode number for evaluation.
        n_episode=5,
        # Evaluate every "eval_freq" training steps.
        eval_freq=10,
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
    commander=dict(),
)
sumo_dqn_default_config = EasyDict(sumo_dqn_default_config)
main_config = sumo_dqn_default_config
