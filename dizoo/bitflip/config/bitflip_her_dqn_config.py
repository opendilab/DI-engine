from easydict import EasyDict

n_bits = 5  # 15 or 20 n_bits can show differences between pure DQN and HER DQN, 5 n_bits for unittest
bitflip_her_dqn_config = dict(
    exp_name='bitflip_{}bit_herdqn_seed0'.format(n_bits),
    env=dict(
        collector_env_num=8,
        evaluator_env_num=16,
        n_bits=n_bits,
        n_evaluator_episode=16,
        stop_value=0.9,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=2 * n_bits,
            action_shape=n_bits,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        # == Different from most DQN algorithms ==
        # If discount_factor(gamma) > 0.9, it would be very difficult to converge
        discount_factor=0.8,
        learn=dict(
            update_per_collect=10,
            # batch_size = episode_size * sample_per_episode
            # You can refer to cfg.other.her to learn about `episode_size` and `sample_per_episode`
            batch_size=128,
            learning_rate=0.0005,
            target_update_freq=500,
        ),
        collect=dict(
            n_episode=8,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000)),
        other=dict(
            # == Different from most DQN algorithms ==
            # Fix epsilon to 0.2 leads to easier convergence, proposed in the paper.
            eps=dict(
                type='exp',
                start=0.2,  # 0.8
                end=0.2,  # original0.1, paper0.15~0.2
                decay=100,  # 10000
            ),
            replay_buffer=dict(replay_buffer_size=4000, ),
            her=dict(
                her_strategy='future',
                # her_replay_k=2,  # `her_replay_k` is not used in episodic HER
                # Sample how many episodes in each train iteration.
                episode_size=32,
                # Generate how many samples from one episode.
                sample_per_episode=4,
            ),
        ),
    ),
)
bitflip_her_dqn_config = EasyDict(bitflip_her_dqn_config)
main_config = bitflip_her_dqn_config

bitflip_her_dqn_create_config = dict(
    env=dict(
        type='bitflip',
        import_names=['dizoo.bitflip.envs.bitflip_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='episode'),
    collector=dict(type='episode'),
)
bitflip_her_dqn_create_config = EasyDict(bitflip_her_dqn_create_config)
create_config = bitflip_her_dqn_create_config

if __name__ == '__main__':
    from dizoo.bitflip.entry.bitflip_dqn_main import main
    main(main_config, seed=0)
