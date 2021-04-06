from easydict import EasyDict

nstep = 2
traj_len = 16
pomdp_dqn_default_config = dict(
    env=dict(
        # Support ['base', 'subprocess']. 'base' is pseudo parallel and 'subprocess' is subprocess parallel.
        # 'base' is used for some light env(e.g. cartpole), whose env step time is much shorter than IPC time.
        # 'subprocess' is used for more complicated env(e.g. pong and larger than pong), which is more recommended to use in practice.
        env_manager_type='subprocess',
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # manager=dict(shared_memory=True, ),
        # Env register name (refer to function "register_env").
        env_type='pomdp',
        # Env number respectively for actor and evaluator.
        actor_env_num=6,
        evaluator_env_num=3,
        # POMDP config
        env_id='Pong-ramNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        warp_frame=False,
        use_ram=True,
        clip_reward=False,
        render=False,
        pomdp=dict(noise_scale=0.01, zero_p=0.2, reward_noise=0.01, duplicate_p=0.2),
    ),
    policy=dict(
        # Whether to use cuda for network.
        use_cuda=False,
        # RL policy register name (refer to function "register_policy").
        policy_type='dqn',
        import_names=['nervex.policy.dqn'],
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # Model config used for model creating. Remember to change this, especially "obs_dim" and "action_dim" according to specific env.
        model=dict(
            obs_dim=(512, ),
            action_dim=6,
            embedding_dim=128,
            # Whether to use dueling head.
            dueling=True,
        ),
        # learn_mode config
        learn=dict(
            # How many steps to train after actor's one collection. Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            train_iteration=20,
            batch_size=32,
            learning_rate=0.001,
            # L2 norm weight for network parameters.
            weight_decay=0.0,
            algo=dict(
                # Frequence of target network update.
                target_update_freq=500,
                # Reward's future discount facotr, aka. gamma.
                discount_factor=0.99,
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
                # type='exp',
                # start=0.95,
                # end=0.05,
                # decay=10000,
                type='exp',
                start=1.,
                end=0.05,
                decay=200000,
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
        n_sample=100,
        # Get "n_episode" complete episodic trajectories per collect.
        # n_episode=8,
        traj_len=traj_len,
        collect_print_freq=100,
    ),
    evaluator=dict(
        # Episode number for evaluation.
        n_episode=3,
        # Evaluate every "eval_freq" training steps.
        eval_freq=1000,
        # Once evaluation reward reaches "stop_value", which means the policy converges, then the whole training can end.
        stop_value=20,
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
                ext_args=dict(freq=1000, ),
            ),
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=1000, ),
            ),
        ),
    ),
    commander=dict(),
)
pomdp_dqn_default_config = EasyDict(pomdp_dqn_default_config)
main_config = pomdp_dqn_default_config
