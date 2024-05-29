from typing import List, Optional
import os

from easydict import EasyDict
from gym.spaces import Space, Discrete
from gym.spaces.box import Box
import gym
import numpy as np
import imageio


from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('taxi')
class TaxiEnv(BaseEnv):
    
    def __init__(self, cfg: dict) -> None:
        
        self._cfg = cfg
        assert self._cfg.env_id == "Taxi-v3", "Your environment name is not Taxi-v3!"
        self._init_flag = False
        self._replay_path = None
        self._save_replay = False
        self._frames = []
       
    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(
                id=self._cfg.env_id,
                render_mode="single_rgb_array",
                max_episode_steps=self._cfg.max_episode_steps
            )
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._reward_space = Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )    
        self._init_flag = True    
        self._eval_episode_return = 0
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env_seed = self._seed + np_seed 
        elif hasattr(self, '_seed'):
            self._env_seed = self._seed
        if hasattr(self, '_seed'):
            obs = self._env.reset(seed=self._env_seed)
        else:
            obs = self._env.reset()
        
        if self._save_replay:
            picture = self._env.render()
            self._frames.append(picture)
        self._eval_episode_return = 0.
        obs = to_ndarray(obs)
        return obs
    
    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False
        
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        action = action.item()
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # Transformed to an array with shape (1, )
        if self._save_replay:
            picture = self._env.render()
            self._frames.append(picture)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay:
                assert self._replay_path is not None, "your should have a path"
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
                )
                self.frames_to_gif(self._frames, path)
                self._frames = []
                self._save_replay_count += 1
        rew = rew.astype(np.float32)
        obs = obs.astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)
    
    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
            if not os.path.exists(replay_path):
                os.makedirs(replay_path)
        self._replay_path = replay_path
        self._save_replay = True
        self._save_replay_count = 0
        
    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        elif isinstance(random_action, dict):
            random_action = to_ndarray(random_action)
        else:
            raise TypeError(
                '`random_action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(
                type(random_action), random_action
                )
            )
        return random_action
        
    #todo encode the state into a vector    
    def _encode_taxi(self, obs: np.ndarray) -> np.ndarray:
        taxi_row, taxi_col, passenger_location, destination = self._env.unwrapped.decode(obs)
        
    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def reward_space(self) -> Space:
        return self._reward_space
    
    def __repr__(self) -> str:
        return "DI-engine Taxi-v3 Env"
    
    @staticmethod
    def frames_to_gif(frames: List[imageio.core.util.Array], gif_path: str, duration: float = 0.1) -> None:
        """
        Convert a list of frames into a GIF.
        Args:
        - frames (List[imageio.core.util.Array]): A list of frames, each frame is an image.
        - gif_path (str): The path to save the GIF file.
        - duration (float): Duration between each frame in the GIF (seconds).

        Returns:
        None, the GIF file is saved directly to the specified path.
        """
        # Save all frames as temporary image files
        temp_image_files = []
        for i, frame in enumerate(frames):
            temp_image_file = f"frame_{i}.png"  # Temporary file name
            imageio.imwrite(temp_image_file, frame)  # Save the frame as a PNG file
            temp_image_files.append(temp_image_file)

        # Use imageio to convert temporary image files to GIF
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for temp_image_file in temp_image_files:
                image = imageio.imread(temp_image_file)
                writer.append_data(image)

        # Clean up temporary image files
        for temp_image_file in temp_image_files:
            os.remove(temp_image_file)
        print(f"GIF saved as {gif_path}")
