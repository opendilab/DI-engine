import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
import numpy as np
import operator
from functools import reduce
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX


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


class MoveBonus(Wrapper):
    """
    Adds an movement bonus based on which positions
    are visited on the grid.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import PositionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = MoveBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited positions.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.goal_pos = (self.width - 2, self.height - 2)
        self.scale = np.sqrt(self.width ** 2 + self.height ** 2)

    def step(self, action):
        """Steps through the environment with `action`."""

        cur_dis = np.sqrt(
            (self.goal_pos[0] - self.env.agent_pos[0]) ** 2 + (self.goal_pos[1] - self.env.agent_pos[1]) ** 2
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        tmp_dis = np.sqrt(
            (self.goal_pos[0] - self.env.agent_pos[0]) ** 2 + (self.goal_pos[1] - self.env.agent_pos[1]) ** 2
        )

        move_bonus = (cur_dis - tmp_dis) / self.scale
        reward += move_bonus

        return obs, reward, terminated, truncated, info


class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import OneHotPartialObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0]], dtype=uint8)
        >>> env = OneHotPartialObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
              dtype=uint8)
    """

    def __init__(self, env):
        """A wrapper that makes the image observation a one-hot encoding of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + 1

        new_image_space = spaces.Box(low=0, high=1, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="float32")
        self.observation_space = spaces.Dict({**self.observation_space.spaces, "image": new_image_space})

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="float32")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class FlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize, ),
            dtype="float32",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        img = obs["image"]

        img = img.flatten()

        return img
