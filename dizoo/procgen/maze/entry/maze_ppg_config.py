from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

maze_ppg_default_config = dict(
    exp_name='maze_ppg2_normT',
    env=dict(
        is_train=True,
        collector_env_num=64,
        evaluator_env_num=10,
        n_evaluator_episode=50,
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
            epoch_per_collect=1,
            value_norm=True,
            batch_size=2048,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(n_sample=16384, ),
        eval=dict(evaluator=dict(eval_freq=24, )),
        other=dict(
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
    env_manager=dict(type='subprocess', ),
    policy=dict(type='ppg'),
)
maze_ppg_create_config = EasyDict(maze_ppg_create_config)
create_config = maze_ppg_create_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
