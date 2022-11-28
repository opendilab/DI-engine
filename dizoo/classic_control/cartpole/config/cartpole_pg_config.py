from easydict import EasyDict

cartpole_pg_config = dict(
    exp_name='cartpole_pg_seed0',
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
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            entropy_weight=0.001,
        ),
        collect=dict(n_episode=80, unroll_len=1, discount_factor=0.9),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
cartpole_pg_config = EasyDict(cartpole_pg_config)
main_config = cartpole_pg_config
cartpole_pg_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
cartpole_pg_create_config = EasyDict(cartpole_pg_create_config)
create_config = cartpole_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
