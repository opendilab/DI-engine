from typing import Any, Dict, Optional
from easydict import EasyDict
import matplotlib.pyplot as plt
import gym
import copy
import numpy as np
from ding.envs.env.base_env import BaseEnvTimestep
from ding.torch_utils.data_helper import to_ndarray
from ding.utils.default_helper import deep_merge_dicts
from dizoo.metadrive.env.drive_utils import BaseDriveEnv


def draw_multi_channels_top_down_observation(obs, show_time=0.5):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()

    timer = fig.canvas.new_timer(interval=show_time * 1000)
    timer.add_callback(close_event)
    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()
    plt.close()


class DriveEnvWrapper(gym.Wrapper):
    """
    Overview:
        Environment wrapper to make ``gym.Env`` align with DI-engine definitions, so as to use utilities in DI-engine.
        It changes ``step``, ``reset`` and ``info`` method of ``gym.Env``, while others are straightly delivered.

    Arguments:
        - env (BaseDriveEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.
    """
    config = dict()

    def __init__(self, env: BaseDriveEnv, cfg: Dict = None, **kwargs) -> None:
        if cfg is None:
            self._cfg = self.__class__.default_config()
        elif 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self.env = env
        if not hasattr(self.env, 'reward_space'):
            self.reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
        if 'show_bird_view' in self._cfg and self._cfg['show_bird_view'] is True:
            self.show_bird_view = True
        else:
            self.show_bird_view = False
        self.action_space = self.env.action_space
        self.env = env

    def reset(self, *args, **kwargs) -> Any:
        """
        Overview:
            Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
            are recorded.
        Returns:
            - Any: Observations from environment
        """
        obs = self.env.reset(*args, **kwargs)
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            birdview = obs['birdview'].transpose((2, 0, 1))
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        self._eval_episode_return = 0.0
        self._arrive_dest = False
        return obs

    def step(self, action: Any = None) -> BaseEnvTimestep:
        """
        Overview:
            Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
            that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
            namedtuple defined in DI-engine. It will also convert actions, observations and reward into
            ``np.ndarray``, and check legality if action contains control signal.
        Arguments:
            - action (Any, optional): Actions sent to env. Defaults to None.
        Returns:
            - BaseEnvTimestep: DI-engine format of env step returns.
        """
        action = to_ndarray(action)
        obs, rew, done, info = self.env.step(action)
        if self.show_bird_view:
            draw_multi_channels_top_down_observation(obs, show_time=0.5)
        self._eval_episode_return += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            birdview = obs['birdview'].transpose((2, 0, 1))
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(obs, rew, done, info)

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, shape=(5, 84, 84), dtype=np.float32)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self.env = gym.wrappers.Monitor(self.env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    def __repr__(self) -> str:
        return repr(self.env)

    def render(self):
        self.env.render()

    def clone(self, caller: str):
        cfg = copy.deepcopy(self._cfg)
        return DriveEnvWrapper(self.env.clone(caller), cfg)
