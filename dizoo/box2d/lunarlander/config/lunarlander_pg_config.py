from easydict import EasyDict

lunarlander_pg_config = dict(
    exp_name='lunarlander_pg_seed0',
    env=dict(
        env_id='LunarLander-v2',
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            batch_size=320,
            learning_rate=3e-4,
            entropy_weight=0.001,
            grad_norm=0.5,
        ),
        collect=dict(n_episode=8, discount_factor=0.99),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
    ),
)
lunarlander_pg_config = EasyDict(lunarlander_pg_config)
main_config = lunarlander_pg_config
lunarlander_pg_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
lunarlander_pg_create_config = EasyDict(lunarlander_pg_create_config)
create_config = lunarlander_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c lunarlander_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0, max_env_step=int(1e7))
