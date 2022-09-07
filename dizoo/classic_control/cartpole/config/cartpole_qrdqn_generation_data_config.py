from easydict import EasyDict

cartpole_qrdqn_generation_data_config = dict(
    exp_name='cartpole_qrdqn_generation_data_seed0',
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
        collect=dict(
            collect_count=1000,
            data_type='hdf5',
            # pretrained RL model path, user can modify it as its own path
            model_path='./cartpole_qrdqn_seed0/ckpt/ckpt_best.pth.tar',
            # this prefix should be the same as exp_name
            save_path='./cartpole_qrdqn_generation_data_seed0/expert.pkl',
        ),
        other=dict(eps=dict(collect=0.2, ), ),
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

if __name__ == "__main__":
    from ding.entry import collect_demo_data
    cfg = main_config.policy.collect
    collect_demo_data(
        (main_config, create_config), seed=0, collect_count=cfg.collect_count, state_dict_path=cfg.model_path
    )
