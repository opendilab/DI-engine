from ding.entry.serial_entry_sil import serial_pipeline_sil
from easydict import EasyDict

cartpole_a2c_config = dict(
    exp_name='cartpole_a2c',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        on_policy=True,
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            batch_size=64,
            # (bool) Whether to normalize advantage. Default to False.
            unroll_len=1,
            normalize_advantage=False,
            learning_rate=0.001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
        ),
        collect=dict(
            collector=dict(
                type='episode',
                get_train_sample=True,
            ),
            # (int) collect n_sample data, train model n_iteration times
            n_episode=8,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
        ),
        other=dict(sil=dict(
            value_weight=0.5,
            learning_rate=0.001,
        ), replay_buffer=dict(replay_buffer_size=20000, )),
    ),
)
cartpole_a2c_config = EasyDict(cartpole_a2c_config)
main_config = cartpole_a2c_config

cartpole_a2c_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='a2c'),
)
cartpole_a2c_create_config = EasyDict(cartpole_a2c_create_config)
create_config = cartpole_a2c_create_config

if __name__ == '__main__':
    from ding.entry import serial_entry_sil

    serial_pipeline_sil((main_config, create_config), seed=0)
