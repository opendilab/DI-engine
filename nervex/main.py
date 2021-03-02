import pytest
import time
import os
from copy import deepcopy
from app_zoo.classic_control.cartpole.entry import cartpole_dqn_default_config

from nervex.entry.serial_entry3 import serial_pipeline


def main():
    config = deepcopy(cartpole_dqn_default_config)
    serial_pipeline(config, seed=0)


if __name__ == "__main__":
    main()