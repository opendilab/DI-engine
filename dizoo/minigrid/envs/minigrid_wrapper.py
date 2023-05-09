import gymnasium as gym
import numpy as np
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


class ObsPlusPrevActRewWrapper(gym.Wrapper):
    """
    Overview:
       This wrapper is used in policy NGU.
       Set a dict {'obs': obs, 'prev_action': self.prev_action, 'prev_reward_extrinsic': self.prev_reward_extrinsic}
       as the new wrapped observation,
       which including the current obs, previous action and previous reward.
    Interface:
        ``__init__``, ``reset``, ``step``
    Properties:
        - env (:obj:`gymnasium.Env`): the environment to wrap.
    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gymnasium.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                'obs': env.observation_space,
                'prev_action': env.action_space,
                'prev_reward_extrinsic': gym.spaces.Box(
                    low=env.reward_range[0], high=env.reward_range[1], shape=(1, ), dtype=np.float32
                )
            }
        )
        self.prev_action = -1  # null action
        self.prev_reward_extrinsic = 0  # null reward

    def reset(self, *, seed: int = None):
        """
        Overview:
            Resets the state of the environment.
        Returns:
            -  obs (:obj:`Dict`) : the wrapped observation, which including the current obs, \
                previous action and previous reward.
        """
        obs, info = self.env.reset(seed=seed)
        obs = {'obs': obs, 'prev_action': self.prev_action, 'prev_reward_extrinsic': self.prev_reward_extrinsic}
        return obs, info

    def step(self, action):
        """
        Overview:
            Step the environment with the given action.
            Save the previous action and reward to be used in next new obs
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            -  obs (:obj:`Dict`) : the wrapped observation, which including the current obs, \
                previous action and previous reward.
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further \
                 step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful  \
                for debugging, and sometimes learning)
        """

        obs, reward, done, truncated, info = self.env.step(action)
        obs = {'obs': obs, 'prev_action': self.prev_action, 'prev_reward_extrinsic': self.prev_reward_extrinsic}
        self.prev_action = action
        self.prev_reward_extrinsic = reward
        return obs, reward, done, truncated, info
