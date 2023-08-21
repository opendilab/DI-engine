from easydict import EasyDict

cartpole_discrete_dt_config = dict(
    exp_name='cartpole_dt_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    dataset=dict(
        data_dir_prefix='./cartpole_qrdqn_generation_data_seed0/expert_demos.hdf5',
        rtg_scale=None,
        context_len=20,
        env_type='classic',
    ),
    policy=dict(
        cuda=False,
        rtg_target=10,
        evaluator_env_num=5,
        clip_grad_norm_p=1.0,
        state_mean=1,
        state_std=0,
        model=dict(
            state_dim=4,
            act_dim=2,
            n_blocks=6,
            h_dim=128,
            context_len=20,
            n_heads=8,
            drop_p=0.1,
            continuous=False,
        ),
        max_timestep=1000,
        discount_factor=0.97,
        nstep=3,
        batch_size=64,
        learning_rate=0.001,
        target_update_freq=100,
        kappa=1.0,
        min_q_weight=4.0,
        collect=dict(
            data_type='hdf5',
            data_path='./cartpole_qrdqn_generation_data_seed0/expert_demos.hdf5',
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
cartpole_discrete_dt_config = EasyDict(cartpole_discrete_dt_config)
main_config = cartpole_discrete_dt_config
cartpole_discrete_dt_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dt'),
)
cartpole_discrete_dt_create_config = EasyDict(cartpole_discrete_dt_create_config)
create_config = cartpole_discrete_dt_create_config
# You can run this config with the entry file like `ding/example/dt.py`
