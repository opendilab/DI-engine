from easydict import EasyDict
from ding.entry import serial_pipeline

memory_len_a2c_config = dict(
    exp_name='memory_len_10_a2c',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=1,
        n_evaluator_episode=100,
        env_id='memory_len/10',
        stop_value=1.,
    ),
    policy=dict(
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=3,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
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
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
memory_len_a2c_config = EasyDict(memory_len_a2c_config)
main_config = memory_len_a2c_config

memory_len_a2c_create_config = dict(
    env=dict(
        type='bsuite',
        import_names=['dizoo.bsuite.envs.bsuite_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='a2c'),
)
memory_len_a2c_create_config = EasyDict(memory_len_a2c_create_config)
create_config = memory_len_a2c_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)