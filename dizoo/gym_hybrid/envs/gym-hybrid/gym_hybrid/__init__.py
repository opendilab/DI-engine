from gym.envs.registration import register

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