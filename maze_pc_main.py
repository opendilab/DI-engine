from dizoo.maze.config.maze_pc_config import main_config, create_config
from ding.entry import serial_pipeline_pc


serial_pipeline_pc([main_config, create_config], seed=0)
