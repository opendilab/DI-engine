from easydict import EasyDict

pendulum_sac_data_genearation_config = dict(
    exp_name='pendulum_sac_data_generation',
    env=dict(
        collector_env_num=10,
        act_scale=True,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        collect=dict(
            n_sample=1000,
            save_path='./pendulum_sac_data_generation/expert.pkl',
            data_type='hdf5',
            state_dict_path='./pendulum_sac_seed0/ckpt/final.pth.tar',
        ),
    ),
)

pendulum_sac_data_genearation_config = EasyDict(pendulum_sac_data_genearation_config)
main_config = pendulum_sac_data_genearation_config

pendulum_sac_data_genearation_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
pendulum_sac_data_genearation_create_config = EasyDict(pendulum_sac_data_genearation_create_config)
create_config = pendulum_sac_data_genearation_create_config
