from easydict import EasyDict

cartpole_bc_config = dict(
    exp_name='cartpole_bc_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        continuous=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.01,
            learner=dict(hook=dict(save_ckpt_after_iter=1000)),
            train_epoch=20,
        ),
        eval=dict(evaluator=dict(eval_freq=40, ))
    ),
)
cartpole_bc_config = EasyDict(cartpole_bc_config)
main_config = cartpole_bc_config
cartpole_bc_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='bc'),
)
cartpole_bc_create_config = EasyDict(cartpole_bc_create_config)
create_config = cartpole_bc_create_config

if __name__ == "__main__":
    # Note: Users need to generate expert data, and save the data to ``expert_data_path``
    from ding.entry import serial_pipeline_bc
    serial_pipeline_bc([main_config, create_config], seed=0, data_path=expert_data_path)
