from easydict import EasyDict

nstep=3
sokoban_dqn_config = dict(
    exp_name = "sokoban_dqn_seed0",
    env = dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        env_id="Sokoban-v0",
        stop_value=2000,
        n_episode=64,
    ),
    policy = dict(
        cuda = True,
        model = dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # command_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        )
    )
)
sokoban_dqn_config = EasyDict(sokoban_dqn_config)
main_config = sokoban_dqn_config
sokoban_dqn_create_config = dict(
    env = dict(
        type = 'sokoban',
        import_names=['dizoo.sokoban.envs.sokoban_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
sokoban_dqn_create_config = EasyDict(sokoban_dqn_create_config)
create_config = sokoban_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c sokoban_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)