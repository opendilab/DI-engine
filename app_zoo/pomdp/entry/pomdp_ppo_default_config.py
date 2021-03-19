from easydict import EasyDict

pomdp_ppo_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.pomdp.envs.atari_env'],
        env_type='pomdp',
        actor_env_num=6,
        evaluator_env_num=3,
        env_id='Pong-ramNoFrameskip-v4',
        frame_stack=4,
        is_train=True,
        warp_frame=False,
        use_ram=True,
        clip_reward=False,
        render=False,
        # render=True,
        pomdp=dict(noise_scale=0.01, zero_p=0.2, reward_noise=0.01, duplicate_p=0.2),
        # pomdp=dict(
        #     noise_scale=0.0,
        #     zero_p=0.0,
        #     duplicate_p=0.0),
    ),
    policy=dict(
        use_cuda=False,
        policy_type='ppo',
        import_names=['nervex.policy.ppo'],
        on_policy=False,
        model=dict(
            obs_dim=(512, ),
            action_dim=6,
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
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=1000,
            max_reuse=100,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=16,
        traj_len=2000,  # cartpole max episode len
        collect_print_freq=100,
    ),
    evaluator=dict(
        n_episode=5,
        eval_freq=200,
        stop_val=20,
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
pomdp_ppo_default_config = EasyDict(pomdp_ppo_default_config)
main_config = pomdp_ppo_default_config
