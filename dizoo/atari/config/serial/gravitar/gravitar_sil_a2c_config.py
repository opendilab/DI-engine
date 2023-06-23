from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
gravitar_sil_config = dict(
    exp_name='gravitar_sil_a2c_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        env_id='GravitarNoFrameskip-v4',
        # 'ALE/gravitarRevenge-v5' is available. But special setting is needed after gym make.
        stop_value=int(1e9),
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        sil_update_per_collect=2,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=18,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        learn=dict(
            batch_size=40,
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
gravitar_sil_config = EasyDict(gravitar_sil_config)
main_config = gravitar_sil_config

gravitar_sil_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sil_a2c'),
)
gravitar_sil_create_config = EasyDict(gravitar_sil_create_config)
create_config = gravitar_sil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c gravitar_sil_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_sil
    serial_pipeline_sil((main_config, create_config), seed=0, max_env_step=int(3e7))
