from easydict import EasyDict

coinrun_ppo_config = dict(
    env=dict(
        is_train=True,
        env_id='coinrun',
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=10,
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        model=dict(
            obs_shape=[3, 64, 64],
            action_space='discrete',
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
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
coinrun_ppo_config = EasyDict(coinrun_ppo_config)
main_config = coinrun_ppo_config

coinrun_ppo_create_config = dict(
    env=dict(
        type='procgen',
        import_names=['dizoo.procgen.envs.procgen_env'],
    ),
    env_manager=dict(type='subprocess', ),
    policy=dict(type='ppo'),
)
coinrun_ppo_create_config = EasyDict(coinrun_ppo_create_config)
create_config = coinrun_ppo_create_config
