from easydict import EasyDict

hopper_pg_config = dict(
    exp_name='hopper_pg_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=11,
            action_shape=3,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.005,
            entropy_weight=0.01,
        ),
        collect=dict(
            n_episode=34,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1, ))
    ),
)
hopper_pg_config = EasyDict(hopper_pg_config)
main_config = hopper_pg_config
hopper_pg_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
hopper_pg_create_config = EasyDict(hopper_pg_create_config)
create_config = hopper_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c hopper_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
