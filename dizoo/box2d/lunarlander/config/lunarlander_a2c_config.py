from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_a2c_config = dict(
    exp_name='lunarlander_a2c_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=evaluator_env_num,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            batch_size=160,
            learning_rate=3e-4,
            entropy_weight=0.001,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=320,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_a2c_config = EasyDict(lunarlander_a2c_config)
main_config = lunarlander_a2c_config
lunarlander_a2c_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
)
lunarlander_a2c_create_config = EasyDict(lunarlander_a2c_create_config)
create_config = lunarlander_a2c_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0, max_env_step=int(1e7))
