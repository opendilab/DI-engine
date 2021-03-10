import pytest
import time
import os
from copy import deepcopy
from app_zoo.classic_control.cartpole.entry import cartpole_ppo_default_config, cartpole_dqn_default_config
from app_zoo.classic_control.pendulum.entry import pendulum_ppo_default_config, pendulum_sac_auto_alpha_config
from nervex.entry.serial_entry3 import serial_pipeline
import app_zoo.mujoco.envs as envs
import app_zoo.mujoco.entry as entry


def main():
    config = deepcopy(pendulum_sac_auto_alpha_config)
    serial_pipeline(config, seed=0)


if __name__ == "__main__":
    main()
