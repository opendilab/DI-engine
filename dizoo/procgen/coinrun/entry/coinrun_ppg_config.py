from easydict import EasyDict

coinrun_ppg_default_config = dict(
    env=dict(
        is_train=True,
        collector_env_num=4,
        evaluator_env_num=10,
        n_evaluator_episode=40,
        stop_value=11,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[3, 64, 64],
            action_shape=15,
            encoder_hidden_size_list=[32, 32, 64],
        ),
        learn=dict(
            learning_rate=0.0001,
            update_per_collect=5,
            batch_size=64,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(
                multi_buffer=True,
                policy=dict(
                    replay_buffer_size=1000,
                    max_use=100,
                ),
                value=dict(
                    replay_buffer_size=1000,
                    max_use=100,
                ),
            ),
        ),
    ),
)
coinrun_ppg_default_config = EasyDict(coinrun_ppg_default_config)
main_config = coinrun_ppg_default_config

coinrun_ppg_create_config = dict(
    env=dict(
        type='coinrun',
        import_names=['dizoo.procgen.coinrun.envs.coinrun_env'],
    ),
    env_manager=dict(type='subprocess', ),
    policy=dict(type='ppg'),
    replay_buffer=dict(
        policy=dict(type='advanced'),
        value=dict(type='advanced'),
    )
)
coinrun_ppg_create_config = EasyDict(coinrun_ppg_create_config)
create_config = coinrun_ppg_create_config
