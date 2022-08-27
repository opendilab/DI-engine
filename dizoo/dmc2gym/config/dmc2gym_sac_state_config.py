from easydict import EasyDict

dmc2gym_sac_config = dict(
    exp_name='dmc2gym_sac_state_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        domain_name="cartpole",
        task_name="swingup",
        frame_skip=8,
        frame_stack=1,
        from_pixels=False,  # state obs
        channels_first=False,  # obs shape (height, width, 3)
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        # for debug
        # collector_env_num=1,
        # evaluator_env_num=1,
        # n_evaluator_episode=1,
        stop_value=1e6,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_type='state',
        cuda=True,
        random_collect_size=10000,
        # random_collect_size=1,  # for debug
        model=dict(
            obs_shape=5,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            ignore_done=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

dmc2gym_sac_config = EasyDict(dmc2gym_sac_config)
main_config = dmc2gym_sac_config

dmc2gym_sac_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),  # for debug
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
dmc2gym_sac_create_config = EasyDict(dmc2gym_sac_create_config)
create_config = dmc2gym_sac_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c dmc2gym_sac_state_config.py -s 0`
#     from ding.entry import serial_pipeline
#     serial_pipeline([main_config, create_config], seed=0)
if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    for seed in [0, 1, 2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        main_config.exp_name = 'dmc2gym_sac_state' + 'seed' + f'{args.seed}'
        serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,
                        max_env_step=int(3e6))