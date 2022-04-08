from easydict import EasyDict

pendulum_cql_config = dict(
    exp_name='pendulum_cql',
    env=dict(
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=128,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-3,
            learning_rate_alpha=1e-3,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            lagrange_thresh=-1.0,
            min_q_weight=5.0,
        ),
        collect=dict(
            data_type='hdf5',
            data_path='./pendulum_sac_data_generation/expert_demos.hdf5',
            collector_logit=False,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)

pendulum_cql_config = EasyDict(pendulum_cql_config)
main_config = pendulum_cql_config

pendulum_cql_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='cql',
        import_names=['ding.policy.cql'],
    ),
)
pendulum_cql_create_config = EasyDict(pendulum_cql_create_config)
create_config = pendulum_cql_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_offline -c pendulum_cql_config.py -s 0`
    from ding.entry import serial_pipeline_offline
    serial_pipeline_offline([main_config, create_config], seed=0)
