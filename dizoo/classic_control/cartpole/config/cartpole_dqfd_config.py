from easydict import EasyDict

cartpole_dqfd_config = dict(
    exp_name='cartpole_dqfd',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            lambda1=1,
            lambda2=3.0,
            # set this to be 0 (L2 loss = 0) with expert_replay_buffer_size = 0 and lambda1 = 0
            # recover the one step pdd dqn
            lambda3=0,
            per_train_iter_k=10,
            expert_replay_buffer_size=10000,  # justify the buffer size of the expert buffer
        ),
        # Users should add their own path here (path should lead to a well-trained model)
        collect=dict(n_sample=8, demonstration_info_path='path'),
        # note: this is the times after which you learns to evaluate
        eval=dict(evaluator=dict(eval_freq=50, )),
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
cartpole_dqfd_config = EasyDict(cartpole_dqfd_config)
main_config = cartpole_dqfd_config
cartpole_dqfd_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqfd'),
)
cartpole_dqfd_create_config = EasyDict(cartpole_dqfd_create_config)
create_config = cartpole_dqfd_create_config
