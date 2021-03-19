from easydict import EasyDict

use_twin_critic = False
halfcheetah_ddpg_default_config = dict(
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=True, ),
        norm_reward=dict(
            use_norm=False,
            reward_discount=0.98,
        ),
        env_manager_type='subprocess',
        import_names=['app_zoo.mujoco.envs.mujoco_env'],
        env_type='mujoco',
        actor_env_num=16,
        evaluator_env_num=8,
        use_act_scale=True,
    ),
    policy=dict(
        use_cuda=True,
        policy_type='ddpg',
        import_names=['nervex.policy.ddpg'],
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=11,
            action_dim=2,
            use_twin_critic=use_twin_critic,
        ),
        learn=dict(
            train_step=2,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            weight_decay=0.0001,
            ignore_done=True,
            algo=dict(
                target_theta=0.005,
                discount_factor=0.99,
                actor_update_freq=1,
                use_twin_critic=use_twin_critic,
                use_noise=True,
                noise_sigma=0.2,
                noise_range=dict(
                    min=-0.5,
                    max=0.5,
                ),
            ),
            init_data_count=5000,
        ),
        collect=dict(
            traj_len=1,
            unroll_len=1,
            algo=dict(noise_sigma=0.1, ),
        ),
        command=dict(),
    ),
    replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=20000,
            max_reuse=16,
            min_sample_ratio=1,
        ),
    ),
    actor=dict(
        n_sample=48,
        traj_len=1,
        collect_print_freq=1000,
    ),
    evaluator=dict(
        n_episode=8,
        eval_freq=1000,
        stop_val=11000,
    ),
    learner=dict(
        hook=dict(
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
halfcheetah_ddpg_default_config = EasyDict(halfcheetah_ddpg_default_config)
main_config = halfcheetah_ddpg_default_config
