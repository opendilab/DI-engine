import gfootball
import gfootball.env as football_env

import copy
from collections import namedtuple
from typing import List, Any, Optional

import numpy as np
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from .action.gfootball_action_runner import GfootballRawActionRunner
from .obs.gfootball_obs_runner import GfootballObsRunner
from .reward.gfootball_reward_runner import GfootballRewardRunner
import gym
from ding.torch_utils import to_ndarray, to_list
import os
from matplotlib import animation
import matplotlib.pyplot as plt
from ding.envs import ObsPlusPrevActRewWrapper


@ENV_REGISTRY.register('gfootball')
class GfootballEnv(BaseEnv):
    timestep = namedtuple('GfootballTimestep', ['obs', 'reward', 'done', 'info'])

    info_template = namedtuple('GFootballEnvInfo', ['obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg):
        self._cfg = cfg
        self._action_helper = GfootballRawActionRunner(cfg)
        self._reward_helper = GfootballRewardRunner(cfg)
        self._obs_helper = GfootballObsRunner(cfg)
        self.save_replay = cfg.get("save_replay", False)
        self._launch_env_flag = False
        self._launch_env()
        self.env_name = self._cfg.env_name
        self._save_replay_gif = self._cfg.save_replay_gif

    def _launch_env(self, gui=False):

        self._env = football_env.create_environment(
            # default env_name="11_vs_11_stochastic",
            env_name=self._cfg.env_name,
            representation='raw',
            stacked=False,
            logdir='./tmp/football',
            write_goal_dumps=False,
            write_full_episode_dumps=self.save_replay,
            write_video=self.save_replay,
            render=False
        )
        self._launch_env_flag = True

    def reset(self) -> dict:
        if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
            # for NGU
            self.prev_action = -1  # null action
            self.prev_reward_extrinsic = 0  # null reward

        if self._save_replay_gif:
            self._frames = []
        if not self._launch_env_flag:
            self._launch_env()
        self._football_obs = self._env.reset()[0]
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self._observation_space = gym.spaces.Dict(
            {
                'match': gym.spaces.Dict(
                    {
                        k: gym.spaces.Discrete(v['max']) if v['dinfo'] == 'one-hot' else
                        gym.spaces.Box(low=np.array(v['min']), high=np.array(v['max']), dtype=np.float32)
                        for k, v in self._obs_helper.info['match'].value.items()
                    }
                ),
                'player': gym.spaces.Dict(
                    {
                        k: gym.spaces.Discrete(v['max']) if v['dinfo'] == 'one-hot' else
                        gym.spaces.Box(low=np.array(v['min']), high=np.array(v['max']), dtype=np.float32)
                        for k, v in self._obs_helper.info['player'].value['players'].items()
                    }
                )
            }
        )
        self._action_space = gym.spaces.Discrete(self._action_helper.info.shape[0])
        self._reward_space = gym.spaces.Box(
            low=self._reward_helper.info.value['min'],
            high=self._reward_helper.info.value['max'],
            shape=self._reward_helper.info.shape,
            dtype=np.float32
        )

        self.obs = self._obs_helper.get(self)

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
            # for NGU
            return {
                'obs': {
                    'processed_obs': self.obs,
                    'raw_obs': self._football_obs
                },
                'prev_action': self.prev_action,
                'prev_reward_extrinsic': self.prev_reward_extrinsic
            }
        else:
            return {'processed_obs': self.obs, 'raw_obs': self._football_obs}

    def step(self, action: np.array) -> 'GfootballEnv.timestep':
        assert self._launch_env_flag
        self.agent_action = action
        action = action.item()
        # env step
        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        self._football_obs, self._reward_of_action, self._is_done, self._info = self._env.step(action)
        self._football_obs = self._football_obs[0]
        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        info = {'cum_reward': self._reward_helper.cum_reward}
        if self._is_done:
            info['final_eval_reward'] = to_ndarray(self._reward_helper.cum_reward)
            if self._save_replay_gif:
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self.env_name, self._save_replay_gif_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_gif_count += 1
                print(f'save one episode replay_gif in {path}')
        # TODO(pu)
        self.reward = to_ndarray(self.reward)

        if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
            # for NGU
            self.prev_action = action
            self.prev_reward_extrinsic = self.reward
            obs = {
                'obs': {
                    'processed_obs': self.obs,
                    'raw_obs': self._football_obs
                },
                'prev_action': self.prev_action,
                'prev_reward_extrinsic': self.prev_reward_extrinsic
            }
        else:
            obs = {'processed_obs': self.obs, 'raw_obs': self._football_obs}

        return GfootballEnv.timestep(obs, reward=self.reward, done=self._is_done, info=info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        self._env.close()

    def __repr__(self) -> str:
        return 'GfootballEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    def info(self) -> 'GfootballEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return GfootballEnv.info_template(**info_data)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.save_replay = False
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.save_replay = True
        return [cfg for _ in range(evaluator_env_num)]

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

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._save_replay_gif = True
        self._replay_path = replay_path
        self._save_replay_gif_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)


GfootballTimestep = GfootballEnv.timestep
