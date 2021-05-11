from easydict import EasyDict

nstep = 1
cartpole_dqn_config = dict(
    seed=0,
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            discount_factor=0.97,
            nstep=nstep,
        ),
        collect=dict(
            nstep=nstep,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=50,
            )
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=20000,
            ),
        ),
    ),
)
cartpole_dqn_config = EasyDict(cartpole_dqn_config)
main_config = cartpole_dqn_config
cartpole_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(
        type='base'
    ),
    policy=dict(
        type='dqn'
    ),
)
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config
