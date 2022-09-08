from typing import Any, List, Union, Optional
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('mountain_car')
class MountainCarEnv(BaseEnv):
    """
    Implementation of DI-engine's version of the Mountain Car deterministic MDP. 

    Important references that contributed to the creation of this env:
    > Source code of OpenAI's mountain car gym : https://is.gd/y1FkMT
    > Gym documentation of mountain car : https://is.gd/29S0dt
    > Based off DI-engine existing implementation of cartpole_env.py
    > DI-engine's env creation conventions : https://is.gd/ZHLISj

    Only __init__ , step, seed and reset are mandatory & impt.
    The other methods are generally for convenience.
    """

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None

        # Following specifications from https://is.gd/29S0dt
        self._observation_space = gym.spaces.Box(
            low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), shape=(2, ), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(3, start=0)
        self._reward_space = gym.spaces.Box(low=-1, high=0.0, shape=(1, ), dtype=np.float32)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        # Instantiate environment if not already done so
        if not self._init_flag:
            self._env = gym.make('MountainCar-v0')
            self._init_flag = True

        # Check if we have a valid replay path and save replay video accordingly
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}'.format(id(self))
            )

        # Set the seeds for randomization.
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)

        # Get first observation from original environment
        obs = self._env.reset()

        # Convert to numpy array as output
        obs = to_ndarray(obs).astype(np.float32)

        # Init final reward : cumulative sum of the real rewards obtained by a whole episode,
        # used to evaluate the agent Performance on this environment, not used for training.
        self._final_eval_reward = 0.
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:

        # Making sure that input action is of numpy ndarray
        assert isinstance(action, np.ndarray), type(action)

        # Extract action as int, 0-dim array
        action = action.squeeze()

        # Take a step of faith into the unknown!
        obs, rew, done, info = self._env.step(action)

        # Cummulate reward
        self._final_eval_reward += rew

        # Save final cummulative reward when done.
        if done:
            info['final_eval_reward'] = self._final_eval_reward

        # Making sure we conform to di-engine conventions
        obs = to_ndarray(obs)
        rew = to_ndarray([rew]).astype(np.float32)

        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        # If init flag is False, then reset() was never run, no point closing.
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Mountain Car Env"
