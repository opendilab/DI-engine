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
        cuda=True,
        continuous=False,
        loss_type='l1_loss',
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        learn=dict(
            train_epoch=20,
            batch_size=128,
            learning_rate=0.001,
            weight_decay=1e-4,
            momentum=0.9,
            decay_epoch=30,
            decay_rate=1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            lr_decay=True,
        ),
        collect=dict(
            n_episode=10,
            # control the number (alpha*n_episode) of post-demonstration environment interactions at each iteration.
            # Notice: alpha * n_episode > collector_env_num
            data_path="abs data path",  # expert data path
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
    ),
)
cartpole_bc_config = EasyDict(cartpole_bc_config)
main_config = cartpole_bc_config
cartpole_bc_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bc'),
)
cartpole_bc_create_config = EasyDict(cartpole_bc_create_config)
create_config = cartpole_bc_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bc
    serial_pipeline_bc(
        [main_config, create_config], seed=0, max_iter=100000
    )
