from easydict import EasyDict

pendulum_a2c_config = dict(
    exp_name='pendulum_a2c_seed0',
    env=dict(
        collector_env_num=10,
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
            epoch_per_collect=10,
            batch_size=32,
            learning_rate=3e-5,
            value_weight=0.5,
            entropy_weight=0.0,
        ),
        collect=dict(
            n_sample=200,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=200, ))
    ),
)
pendulum_a2c_config = EasyDict(pendulum_a2c_config)
main_config = pendulum_a2c_config
pendulum_a2c_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='a2c'),
)
pendulum_a2c_create_config = EasyDict(pendulum_a2c_create_config)
create_config = pendulum_a2c_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c pendulum_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
