from easydict import EasyDict

cartpole_discrete_cql_config = dict(
    exp_name='cartpole_cql_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=64,
        ),
        discount_factor=0.97,
        nstep=3,
        learn=dict(
            train_epoch=3000,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
            kappa=1.0,
            min_q_weight=4.0,
        ),
        collect=dict(
            data_type='hdf5',
            # offline data path
            data_path='./cartpole_qrdqn_generation_data_seed0/expert_demos.hdf5',
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
cartpole_discrete_cql_config = EasyDict(cartpole_discrete_cql_config)
main_config = cartpole_discrete_cql_config
cartpole_discrete_cql_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='cql_discrete'),
)
cartpole_discrete_cql_create_config = EasyDict(cartpole_discrete_cql_create_config)
create_config = cartpole_discrete_cql_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_offline -c cartpole_cql_config.py -s 0`
    from ding.entry import serial_pipeline_offline
    serial_pipeline_offline((main_config, create_config), seed=0)
