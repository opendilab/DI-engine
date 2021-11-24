from easydict import EasyDict

cartpole_qrdqn_generation_data_config = dict(
    exp_name='cartpole_generation',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        priority=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=64,
        ),
        discount_factor=0.97,
        nstep=3,
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
            kappa=1.0,
            learner=dict(
                load_path='./cartpole/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='./cartpole/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                ),
            ),
        ),
        collect=dict(
            n_sample=80,
            unroll_len=1,
            data_type='hdf5',
            save_path='./cartpole_generation/expert.pkl',
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
                collect=0.2,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
cartpole_qrdqn_generation_data_config = EasyDict(cartpole_qrdqn_generation_data_config)
main_config = cartpole_qrdqn_generation_data_config
cartpole_qrdqn_generation_data_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='qrdqn'),
)
cartpole_qrdqn_generation_data_create_config = EasyDict(cartpole_qrdqn_generation_data_create_config)
create_config = cartpole_qrdqn_generation_data_create_config
