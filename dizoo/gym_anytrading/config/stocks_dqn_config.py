from easydict import EasyDict

stocks_dqn_config = dict(
    exp_name='stocks_dqn_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='stocks-v0',
        n_evaluator_episode=8,
        stop_value=2,
        # one trading year.
        eps_length=253,
        # associated with the feature length.
        window_size=20,
        # the path to save result image.
        save_path='./fig/',
        # the raw data file name
        stocks_data_filename='STOCKS_GOOGL',
        # the stocks range percentage used by train/test.
        # if one of them is None, train & test set will use all data by default.
        train_range=None,
        test_range=None,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            obs_shape=62,
            action_shape=5,
            encoder_hidden_size_list=[128],
            head_layer_num=1,
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=5,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
            ignore_done=True,
        ),
        # collect_mode config
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
        ),
    ),
)
stocks_dqn_config = EasyDict(stocks_dqn_config)
main_config = stocks_dqn_config

stocks_dqn_create_config = dict(
    env=dict(
        type='stocks-v0',
        import_names=['dizoo.gym_anytrading.envs.stocks_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='dqn',
    ),
    evaluator=dict(
        type='trading_interaction',
        import_names=['dizoo.gym_anytrading.worker'],
        ),
)
stocks_dqn_create_config = EasyDict(stocks_dqn_create_config)
create_config = stocks_dqn_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e7))
