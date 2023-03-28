from gym.envs.registration import register
from gym_hybrid.environments import MovingEnv
from gym_hybrid.environments import SlidingEnv
from gym_hybrid.environments import HardMoveEnv

register(
    id='Moving-v0',
    entry_point='gym_hybrid:MovingEnv',
)
register(
    id='Sliding-v0',
    entry_point='gym_hybrid:SlidingEnv',
)
register(
    id='HardMove-v0',
    entry_point='gym_hybrid:HardMoveEnv',
)
