from easydict import EasyDict

cartpole_trex_dqn_config = dict(
    exp_name='cartpole_trex_dqn',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn/video',
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='dqn',
        env_id='CartPole-v0',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=500,
        checkpoint_step=100,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./cartpole.params',
        offline_data_path='abs data path',
    ),
    policy=dict(
        load_path='',
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
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=40, )),
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
cartpole_trex_dqn_config = EasyDict(cartpole_trex_dqn_config)
main_config = cartpole_trex_dqn_config
cartpole_trex_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
cartpole_trex_dqn_create_config = EasyDict(cartpole_trex_dqn_create_config)
create_config = cartpole_trex_dqn_create_config
