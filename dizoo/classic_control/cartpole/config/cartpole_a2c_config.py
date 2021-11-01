# ding -m serial_onpolicy -c cartpole_a2c_config.py -s 0
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
