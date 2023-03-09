from easydict import EasyDict

maze_size = 16
num_actions = 4
maze_pc_config = dict(
    exp_name="maze_bc_seed0",
    env=dict(
        collector_env_num=1,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        env_id='Maze',
        size=maze_size,
        wall_type='tunnel',
        stop_value=1
    ),
    policy=dict(
        cuda=True,
        maze_size=maze_size,
        num_actions=num_actions,
        max_bfs_steps=100,
        model=dict(
            obs_shape=[3, maze_size, maze_size],
            action_shape=num_actions,
            encoder_hidden_size_list=[
                128,
                256,
                512,
                1024,
            ],
            strides=[1, 1, 1, 1]
        ),
        learn=dict(
            # update_per_collect=4,
            batch_size=256,
            learning_rate=0.005,
            train_epoch=5000,
            optimizer='SGD',
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
    policy=dict(type='bc'),
)
maze_pc_create_config = EasyDict(maze_pc_create_config)
create_config = maze_pc_create_config

# You can run `dizoo/maze/entry/maze_bc_main.py` to run this config.
