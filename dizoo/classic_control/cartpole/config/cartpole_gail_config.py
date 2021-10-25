from easydict import EasyDict

cartpole_gail_config = dict(
    exp_name='cartpole_gail_train',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='gail',
        input_size=5,
        hidden_size=64,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        expert_data_path='cartpole_dqn/expert_data_train.pkl',
        collect_count=1000,
    ),
    policy=dict(
        load_path='cartpole_gail/reward_model/ckpt/last.pth.tar',
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=64),
        eval=dict(evaluator=dict(eval_freq=10, )),
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
cartpole_gail_config = EasyDict(cartpole_gail_config)
main_config = cartpole_gail_config
cartpole_gail_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
cartpole_gail_create_config = EasyDict(cartpole_gail_create_config)
create_config = cartpole_gail_create_config
