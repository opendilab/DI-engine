
from easydict import EasyDict
from ding.entry import serial_pipeline
from dizoo.classic_control.pendulum.config.pendulum_acer_config import pendulum_acer_config, pendulum_acer_create_config

if __name__ == "__main__":
    serial_pipeline([pendulum_acer_config, pendulum_acer_create_config], seed=0)