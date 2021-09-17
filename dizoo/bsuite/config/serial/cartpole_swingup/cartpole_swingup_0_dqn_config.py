from easydict import EasyDict

cartpole_swingup_0_dqn_config = dict(
    exp_name='cartpole_swingup_0_dqn',
    env=dict(
        collector_env_num=16,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        env_id='cartpole_swingup/0',
        stop_value=100,
    ),
    policy=dict(
        load_path='',
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=200, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
cartpole_swingup_0_dqn_config = EasyDict(cartpole_swingup_0_dqn_config)
main_config = cartpole_swingup_0_dqn_config
cartpole_swingup_0_dqn_create_config = dict(
    env=dict(
        type='bsuite',
        import_names=['dizoo.bsuite.envs.bsuite_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
cartpole_swingup_0_dqn_create_config = EasyDict(cartpole_swingup_0_dqn_create_config)
create_config = cartpole_swingup_0_dqn_create_config
