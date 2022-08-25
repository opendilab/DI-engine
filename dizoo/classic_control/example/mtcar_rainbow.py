from easydict import EasyDict

from dizoo.classic_control.mountain_car.envs.mtcar_env import MountainCarEnv
from dizoo.common.policy.md_rainbow_dqn import MultiDiscreteRainbowDQNPolicy
from ding.policy import RainbowDQNPolicy

from ding.data import DequeBuffer

def main():

    # Init discrete mountain car environment
    env = MountainCarEnv()

    # Config for the case of rainbow and mountain car

    # Since discrete mountain car has 3 actions, we use a multi-discrete version of Rainbow
    policy = MultiDiscreteRainbowDQNPolicy()

    return

if __name__ == '__main__':
    main()