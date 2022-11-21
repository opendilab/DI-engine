from easydict import EasyDict

gym_hybrid_pdqn_config = dict(
    exp_name='gym_hybrid_pdqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1.8,
        save_replay_gif=False,
        replay_path_gif=None,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        discount_factor=0.99,
        nstep=1,
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=500,  # 10~500
            batch_size=320,
            learning_rate_dis=3e-4,
            learning_rate_cont=3e-4,
            target_theta=0.001,
            update_circle=10,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=3200,  # 128,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            noise_sigma=0.1,  # 0.05,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1,
                end=0.1,
                # (int) Decay length(env step)
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
        ),
    )
)

gym_hybrid_pdqn_config = EasyDict(gym_hybrid_pdqn_config)
main_config = gym_hybrid_pdqn_config

gym_hybrid_pdqn_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pdqn'),
)
gym_hybrid_pdqn_create_config = EasyDict(gym_hybrid_pdqn_create_config)
create_config = gym_hybrid_pdqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gym_hybrid_pdqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
