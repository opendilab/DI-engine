from easydict import EasyDict

maze_size = 16
num_actions = 4
maze_pc_config = dict(
    exp_name="maze_pc_seed0",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=1,
        env_id='Maze',
        size=maze_size,
        wall_type='tunnel',
        # max_step=300,
        # stop_value=0.96,
    ),
    policy=dict(
        cuda=True,
        maze_size=maze_size,
        num_actions=num_actions,
        max_bfs_steps=100,
        model=dict(
            obs_shape=[8, maze_size, maze_size],
            action_shape=num_actions,
            encoder_hidden_size_list=[128, 128, 256, 256],
        ),
        learn=dict(
            # update_per_collect=4,
            batch_size=256,
            learning_rate=0.0005,
            train_epoch=40,
            optimizer='SGD',
            # value_weight=0.5,
            # entropy_weight=0.001,
            # clip_ratio=0.2,
            # adv_norm=False,
        ),
        eval=dict(
            evaluator=dict(n_episode=3, stop_value=10000)
        ),
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
    policy=dict(type='pc'),
)
maze_pc_create_config = EasyDict(maze_pc_create_config)
create_config = maze_pc_create_config

# if __name__ == "__main__":
    # or you can enter `ding -m serial -c minigrid_offppo_config.py -s 0`
    # from ding.entry import serial_pipeline
    # serial_pipeline([main_config, create_config], seed=0)
