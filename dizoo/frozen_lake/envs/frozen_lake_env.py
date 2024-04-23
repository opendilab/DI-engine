from typing import Any, Dict, List, Optional
import imageio
import os
import gymnasium as gymn
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('frozen_lake')
class FrozenLakeEnv(BaseEnv):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        assert self._cfg.env_id == "FrozenLake-v1", "yout name is not FrozernLake_v1"
        self._init_flag = False
        self._save_replay_bool = False
        self._save_replay_count = 0
        self._init_flag = False
        self._frames = []
        self._replay_path = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            if not self._cfg.desc:  #specify maps non-preloaded maps
                self._env = gymn.make(
                    self._cfg.env_id,
                    desc=self._cfg.desc,
                    map_name=self._cfg.map_name,
                    is_slippery=self._cfg.is_slippery,
                    render_mode="rgb_array"
                )
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._reward_space = gymn.spaces.Box(
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
            obs, info = self._env.reset(seed=self._env_seed)
        else:
            obs, info = self._env.reset()
        obs = np.eye(16, dtype=np.float32)[obs - 1]
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Dict) -> BaseEnvTimestep:
        obs, rew, terminated, truncated, info = self._env.step(action[0])
        self._eval_episode_return += rew
        obs = np.eye(16, dtype=np.float32)[obs - 1]
        rew = to_ndarray([rew])
        if self._save_replay_bool:
            picture = self._env.render()
            self._frames.append(picture)
        if terminated or truncated:
            done = True
        else:
            done = False
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_bool:
                assert self._replay_path is not None, "your should have a path"
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
                )
                self.frames_to_gif(self._frames, path)
                self._frames = []
                self._save_replay_count += 1
        rew = rew.astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> Dict:
        raw_action = self._env.action_space.sample()
        my_type = type(self._env.action_space)
        return [raw_action]

    def __repr__(self) -> str:
        return "DI-engine Frozen Lake Env"

    @property
    def observation_space(self) -> gymn.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gymn.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gymn.spaces.Space:
        return self._reward_space

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._save_replay_bool = True
        self._save_replay_count = 0
        self._frames = []

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
