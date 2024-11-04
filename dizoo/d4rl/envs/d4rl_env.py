from typing import Any, Union, List
import copy
import numpy as np
import gym
import matplotlib.pyplot as plt
import einops
import imageio
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from .d4rl_wrappers import wrap_d4rl
from ding.utils import ENV_REGISTRY

MAZE_BOUNDS = {'maze2d-umaze-v1': (0, 5, 0, 5), 'maze2d-medium-v1': (0, 8, 0, 8), 'maze2d-large-v1': (0, 9, 0, 12)}


def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))


def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)


def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs


@ENV_REGISTRY.register('d4rl')
class D4RLEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._init_flag = False
        if 'maze' in self._cfg.env_id:
            self.observations = []
            self._extent = (0, 1, 1, 0)

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env(only_info=False)
            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
            self._observation_space = self._env.observation_space
            if 'maze' in self._cfg.env_id:
                new_low = np.tile(self._observation_space.low, 2)
                new_high = np.tile(self._observation_space.high, 2)
                self._observation_space = gym.spaces.Box(low=new_low, high=new_high)
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if 'maze' in self._cfg.env_id:
            target = self._env.get_target()
            self.target_obs = np.array([*target, 0, 0])
        obs = self._env.reset()
        if 'maze' in self._cfg.env_id:
            self.observations.append(obs)
            obs = np.hstack((obs, self.target_obs))
        obs = to_ndarray(obs).astype('float32')
        self._eval_episode_return = 0.
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = {'min': self.action_space.low[0], 'max': self.action_space.high[0], 'dtype': np.float32}
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        if 'maze' in self._cfg.env_id:
            self.observations.append(obs)
            obs = np.hstack([obs, self.target_obs])
        obs = to_ndarray(obs).astype('float32')
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            # self.composite('/mnt/PD/render/rollout.png',self.observations,ncol=1)
        return BaseEnvTimestep(obs, rew, done, info)

    def renders(self, observations, conditions=None, title=None):
        bounds = MAZE_BOUNDS[self._cfg.env_id]

        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self._cfg.env_id}: {bounds}')

        if conditions is not None:
            conditions /= scale

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5, extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0, 1, path_length))
        plt.plot(observations[:, 1], observations[:, 0], c='black', zorder=10)
        plt.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

    def _make_env(self, only_info=False):
        return wrap_d4rl(
            self._cfg.env_id,
            norm_obs=self._cfg.get(
                'norm_obs',
                EasyDict(use_norm=False, offline_stats=dict(use_offline_stats=False, )),
            ),
            norm_reward=self._cfg.get('norm_reward', EasyDict(use_norm=False, )),
            only_info=only_info
        )

    def __repr__(self) -> str:
        return "DI-engine D4RL Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.get('norm_reward', EasyDict(use_norm=False, )).use_norm = False
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
