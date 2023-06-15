from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_sil_config = dict(
    exp_name='lunarlander_sil_a2c_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=evaluator_env_num,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        sil_update_per_collect=1,
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
lunarlander_sil_config = EasyDict(lunarlander_sil_config)
main_config = lunarlander_sil_config

lunarlander_sil_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sil_a2c'),
)
lunarlander_sil_create_config = EasyDict(lunarlander_sil_create_config)
create_config = lunarlander_sil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c lunarlander_sil_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_sil
    serial_pipeline_sil((main_config, create_config), seed=0)
