from easydict import EasyDict
from ding.entry import serial_pipeline_for_anytrading

stocks_a2c_config = dict(
    exp_name='stocks_test_v32',
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
        stocks_data_filename = 'STOCKS_GOOGL',
        # the stocks range percentage used by train/test
        train_range = 0.8,
        test_range = -0.2,
    ),
    policy=dict(
        cuda=True,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=62,
            action_shape=5,
            encoder_hidden_size_list=[128, 64],
        ),
        learn=dict(
            batch_size=64,
            # (bool) Whether to normalize advantage. Default to False.
            normalize_advantage=False,
            learning_rate=0.001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=80,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
    ),
)
stocks_a2c_config = EasyDict(stocks_a2c_config)
main_config = stocks_a2c_config

stocks_a2c_create_config = dict(
    env=dict(
        type='stocks-v0',
        import_names=['dizoo.gym_anytrading.envs.stocks_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='a2c'),
)
stocks_a2c_create_config = EasyDict(stocks_a2c_create_config)
create_config = stocks_a2c_create_config

if __name__ == "__main__":
    serial_pipeline_for_anytrading([main_config, create_config], seed=0, max_env_step=int(1e7))
