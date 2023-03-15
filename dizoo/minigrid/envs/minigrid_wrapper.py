import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper


class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8")

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict({**self.observation_space.spaces, "image": new_image_space})

    def observation(self, obs):
        env = self.unwrapped
        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        # print('grid:' + grid)
        # print('vis_mask:' + vis_mask)
        image = grid.encode(vis_mask)
        return {**obs, "image": image}
