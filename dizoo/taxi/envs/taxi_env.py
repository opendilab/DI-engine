import copy
import os, sys
from typing import List, Tuple, Union, Literal, Optional

import imageio
import gym
from gym.spaces import Space, Discrete
from gym.spaces.box import Box
import numpy as np
from easydict import EasyDict

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('taxi')
class TaxiV3Env(BaseEnv):
    def __init__(self, cfg: dict) -> None:
        
        #^ 该部分为初始化定义，需要有的
        self._cfg = EasyDict(
            env_id='Taxi-v3',
            render_mode='single_rgb_array',
            max_episode_steps=300,  # default max trajectory length to truncate possible infinite attempts
        )
        self._cfg.update(cfg)
        self._env = gym.make(
                "Taxi-v3", render_mode=self._cfg.render_mode, max_episode_steps=self._cfg.max_episode_steps
            )
        self._init_flag = False
        
        #^ SAR space 定义
        self._observation_space = Box(low=0, high=1, shape=(500, ), dtype=np.float32)
        self._action_space = Discrete(6)
        self._reward_space =  Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )
        
        #^ 可视化设定
        self._replay_path = None
        self._save_replay_bool = False
        self._frames = []
       
    
    def reset(self) -> np.ndarray:
        
        if not self._init_flag:
            self._env = gym.make(
                "Taxi-v3", render_mode=self._cfg.render_mode, max_episode_steps=self._cfg.max_episode_steps
            )
            self._init_flag = True
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(500, ), dtype=np.float32)
        self._action_space = Discrete(6)
        self._reward_space = Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )
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
        
        #? 这块多加一点是为了把初始化的首帧也放进去
        if self._save_replay_bool:
            picture = self._env.render()
            self._frames.append(picture)
        self._eval_episode_return = 0.
        obs = to_ndarray(obs)
        return obs
    
    #* 本部分和规范保持大差不差
    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False
        
    #* 本部分和规范保持大差不差
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        
    #* 本部分参考规范，加入gif部分参考了Frozenlake
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        action = action.item()
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # Transformed to an array with shape (1, )
        if self._save_replay_bool:
            picture = self._env.render()
            self._frames.append(picture)
            
        #^ 这里为可视化save过程
        #^ 测试里随机采样木有等到done触发，故可能没有gif保存
        #^ 但当把下面的replay取出来时发现可执行，有gif图片保存。
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
        obs = obs.astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)
    
    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
            if not os.path.exists(replay_path):
                os.makedirs(replay_path)
        self._replay_path = replay_path
        self._save_replay_bool = True
        self._save_replay_count = 0
        
    #* 该部分为random_action 部分，一致
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
        
    #todo 有关taxi的state的编码implementation     
    def _encode_taxi(self, obs: np.ndarray) -> np.ndarray:
        taxi_row, taxi_col, passenger_location, destination = self._env.unwrapped.decode(obs)
        
    #* 三个Property部分，一致
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
        &Convert a list of frames into a GIF.
        *Args:
        *- frames (List[imageio.core.util.Array]): A list of frames, each frame is an image.
        *- gif_path (str): The path to save the GIF file.
        *- duration (float): Duration between each frame in the GIF (seconds).

        ?Returns:
        ?None, the GIF file is saved directly to the specified path.
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
