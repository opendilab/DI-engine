from easydict import EasyDict

pendulum_pg_config = dict(
    exp_name='pendulum_pg_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-200,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=3,
            action_shape=1,
        ),
        learn=dict(
            batch_size=400,
            learning_rate=0.001,
            entropy_weight=0.001,
        ),
        collect=dict(
            n_episode=2,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=200, ))
    ),
)
pendulum_pg_config = EasyDict(pendulum_pg_config)
main_config = pendulum_pg_config
pendulum_pg_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
pendulum_pg_create_config = EasyDict(pendulum_pg_create_config)
create_config = pendulum_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c pendulum_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
