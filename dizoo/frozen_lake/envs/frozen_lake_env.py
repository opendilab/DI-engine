from typing import Any, Dict,List, Optional
import imageio
import os
import gymnasium as gymn
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from easydict import EasyDict
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('frozen_lake')
class FrozenLakeEnv(BaseEnv):
    @classmethod
    # def default_config(cls: type) -> EasyDict:
    #     cfg = EasyDict(copy.deepcopy(cls.config))
    #     cfg.cfg_type = cls.__name__ + 'Dict'
    #     return cfg
    
    def __init__(self,cfg)->None:
        self._cfg=cfg
        assert self._cfg.env_id == "FrozenLake-v1", "yout name is not FrozernLake_v1"
        self._init_flag = False
        
        self._replay_path = self._cfg.save_replay_path
        self._save_replay_gif = self._cfg.save_replay_gif

        self._save_replay_count = 0
        self._init_flag = False
        self._frames = []

    def reset(self)-> np.ndarray:
        if not self._init_flag:
            if not self._cfg.desc :#specify maps non-preloaded maps
                self._env = gymn.make(self._cfg.env_id,
                                      desc=self._cfg.desc,
                                      map_name=self._cfg.map_name,
                                      is_slippery=self._cfg.is_slippery,
                                      render_mode="rgb_array")
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._reward_space = gymn.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
        self._init_flag = True
        self._eval_episode_return = 0
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env_seed=self._seed + np_seed
        elif hasattr(self, '_seed'):
            self._env_seed=self._seed

        if hasattr(self, '_seed'):
            obs,info = self._env.reset(seed=self._env_seed)
        else:
            obs,info = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
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
        action=action['action_args']
        obs, rew, terminated, truncated,info = self._env.step(action)
        self._eval_episode_return += rew
        obs = np.array([obs])
        rew = to_ndarray([rew])
        if self._save_replay:
            picture=self._env.render()
            self._frames.append(picture)
        if terminated or truncated:
            done = True
        else :
            done = False
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay:
                assert self._replay_path is not None
                if not os.path.exists(self._replay_path):
                    os.makedirs(self._replay_path)
                path = os.path.join(
                self._replay_path, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
            )
                self.frames_to_gif(self,self._frames,path)
                self._frames = []
                self._save_replay_count += 1
        return BaseEnvTimestep(obs, rew, done, info)
    
    def random_action(self) -> Dict:
        raw_action = self._env.action_space.sample()
        my_type = type(self._env.action_space)
        return {'action_type': my_type, 'action_args': raw_action}

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
        self._save_replay = True
        self._save_replay_count = 0
        self._frames = []

    @staticmethod
    def frames_to_gif(self,frames, gif_path, duration=0.1):
        """
        将帧列表转换为GIF。

        参数：
        - frames: 帧列表，每个元素是一帧图像。
        - gif_path: GIF 文件保存路径。
        - duration: GIF 每帧之间的持续时间（秒）。

        返回：
        无，直接将 GIF 文件保存到指定路径。
        """
        # 保存所有帧为临时图像文件
        temp_image_files = []
        for i, frame in enumerate(frames):
            temp_image_file = f"frame_{i}.png"  # 临时文件名
            imageio.imwrite(temp_image_file, frame)  # 保存帧为PNG文件
            temp_image_files.append(temp_image_file)

        # 使用imageio将临时图像文件转换为GIF
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for temp_image_file in temp_image_files:
                image = imageio.imread(temp_image_file)
                writer.append_data(image)

        # 清理临时图像文件
        for temp_image_file in temp_image_files:
            os.remove(temp_image_file)

        print(f"GIF已保存为 {gif_path}")
