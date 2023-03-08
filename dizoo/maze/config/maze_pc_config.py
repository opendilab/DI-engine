from easydict import EasyDict

maze_size = 16
num_actions = 4
maze_pc_config = dict(
    exp_name="maze_pc_seed0",
    train_seeds=5,
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        env_id='Maze',
        size=maze_size,
        wall_type='tunnel',
        stop_value=1,
    ),
    policy=dict(
        cuda=True,
        maze_size=maze_size,
        num_actions=num_actions,
        max_bfs_steps=100,
        model=dict(
            obs_shape=[8, maze_size, maze_size],
            action_shape=num_actions,
            encoder_hidden_size_list=[
                128,
                256,
                512,
                1024,
            ],
        ),
        learn=dict(
            batch_size=32,
            learning_rate=0.0005,
            train_epoch=100,
            optimizer='Adam',
        ),
        eval=dict(evaluator=dict(n_episode=5)),
        collect=dict(),
    ),
)
maze_pc_config = EasyDict(maze_pc_config)
main_config = maze_pc_config
maze_pc_create_config = dict(
    env=dict(
        type='maze',
        import_names=['dizoo.maze.envs.maze_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pc_bfs'),
)
maze_pc_create_config = EasyDict(maze_pc_create_config)
create_config = maze_pc_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline_pc
    serial_pipeline_pc([maze_pc_config, maze_pc_create_config], seed=0)
