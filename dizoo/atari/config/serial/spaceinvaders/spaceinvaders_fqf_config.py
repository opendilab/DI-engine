from easydict import EasyDict

spaceinvaders_fqf_config = dict(
    exp_name='spaceinvaders_fqf_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='SpaceInvaders-v4',
        #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            num_quantiles=32,
            quantile_embedding_size=64,
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate_fraction=2.5e-9,
            learning_rate_quantile=0.00005,
            target_update_freq=500,
            ent_coef=0,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.01,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=1000000, ),
        ),
    ),
)
spaceinvaders_fqf_config = EasyDict(spaceinvaders_fqf_config)
main_config = spaceinvaders_fqf_config
spaceinvaders_fqf_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='fqf'),
)
spaceinvaders_fqf_create_config = EasyDict(spaceinvaders_fqf_create_config)
create_config = spaceinvaders_fqf_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c spaceinvaders_fqf_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)