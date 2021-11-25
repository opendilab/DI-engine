from ding.entry.serial_entry_onpolicy import serial_pipeline_onpolicy
from easydict import EasyDict

lunarlander_a2c_config = dict(
    exp_name='lunarlander_a2c_seed0',
    env=dict(
        collector_env_num=4,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 64],
            share_encoder=False,
        ),
        learn=dict(
            batch_size=64,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.1,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.00001,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=64,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.995,
        ),
    ),
)
lunarlander_a2c_config = EasyDict(lunarlander_a2c_config)
main_config = lunarlander_a2c_config

lunarlander_a2c_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
)
lunarlander_a2c_create_config = EasyDict(lunarlander_a2c_create_config)
create_config = lunarlander_a2c_create_config

if __name__ == '__main__':
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
