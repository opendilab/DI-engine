from easydict import EasyDict

maze_ppg_default_config = dict(
    exp_name='maze_pgg_seed0',
    env=dict(
        is_train=True,
        collector_env_num=16,
        evaluator_env_num=10,
        n_evaluator_episode=40,
        stop_value=10,
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
            update_per_collect=4,
            batch_size=256,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(n_sample=1024, ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            replay_buffer=dict(
                multi_buffer=True,
                policy=dict(
                    replay_buffer_size=10000,
                    max_use=3,
                ),
                value=dict(
                    replay_buffer_size=20000,
                    max_use=10,
                ),
            ),
        ),
    ),
)
maze_ppg_default_config = EasyDict(maze_ppg_default_config)
main_config = maze_ppg_default_config

maze_ppg_create_config = dict(
    env=dict(
        type='maze',
        import_names=['dizoo.procgen.maze.envs.maze_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppg'),
    replay_buffer=dict(
        policy=dict(type='advanced'),
        value=dict(type='advanced'),
    )
)
maze_ppg_create_config = EasyDict(maze_ppg_create_config)
create_config = maze_ppg_create_config
# use maze_ppg_main.py as entry
