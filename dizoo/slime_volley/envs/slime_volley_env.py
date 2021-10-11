from namedlist import namedlist
import numpy as np
import gym
from typing import Any, Union, List, Optional
import copy
import slimevolleygym

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray


class GymSelfPlayMonitor(gym.wrappers.Monitor):

    def step(self, *args, **kwargs):
        self._before_step(*args, **kwargs)
        observation, reward, done, info = self.env.step(*args, **kwargs)
        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def _before_step(self, *args, **kwargs):
        if not self.enabled:
            return
        self.stats_recorder.before_step(args[0])


@ENV_REGISTRY.register('slime_volley')
class SlimeVolleyEnv(BaseEnv):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        # agent_vs_bot env is single-agent env. obs, action, done, info are all single.
        # agent_vs_agent env is double-agent env, obs, action, info are double, done is still single.
        self._agent_vs_agent = cfg.agent_vs_agent

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def step(self, action: Union[np.ndarray, List[np.ndarray]]):
        if self._agent_vs_agent:
            assert isinstance(action, list) and isinstance(action[0], np.ndarray)
            action1, action2 = action[0], action[1]
        else:
            assert isinstance(action, np.ndarray)
            action1, action2 = action, None
        assert isinstance(action1, np.ndarray), type(action1)
        assert action2 is None or isinstance(action1, np.ndarray), type(action2)
        if action1.shape == (1, ):
            action1 = action1.squeeze()  # 0-dim tensor
        if action2 is not None and action2.shape == (1, ):
            action2 = action2.squeeze()  # 0-dim tensor
        action1 = SlimeVolleyEnv._process_action(action1)
        action2 = SlimeVolleyEnv._process_action(action2)
        obs1, rew, done, info = self._env.step(action1, action2)
        obs1 = to_ndarray(obs1).astype(np.float32)
        self._final_eval_reward += rew
        # info ('ale.lives', 'ale.otherLives', 'otherObs', 'state', 'otherState')
        if self._agent_vs_agent:
            info = [
                {
                    'ale.lives': info['ale.lives'],
                    'state': info['state']
                }, {
                    'ale.lives': info['ale.otherLives'],
                    'state': info['otherState'],
                    'obs': info['otherObs']
                }
            ]
            if done:
                info[0]['final_eval_reward'] = self._final_eval_reward
                info[1]['final_eval_reward'] = -self._final_eval_reward
        else:
            if done:
                info['final_eval_reward'] = self._final_eval_reward
        reward = to_ndarray([rew]).astype(np.float32)
        if self._agent_vs_agent:
            obs2 = info[1]['obs']
            obs2 = to_ndarray(obs2).astype(np.float32)
            observations = np.stack([obs1, obs2], axis=0)
            rewards = to_ndarray([rew, -rew]).astype(np.float32)
            rewards = rewards[..., np.newaxis]
            return BaseEnvTimestep(observations, rewards, done, info)
        else:
            return BaseEnvTimestep(obs1, reward, done, info)

    def reset(self):
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id)
            if self._replay_path is not None:
                self._env = GymSelfPlayMonitor(
                    self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
                )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        if self._agent_vs_agent:
            obs = np.stack([obs, obs], axis=0)
            return obs
        else:
            return obs

    def info(self):
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=2,
            obs_space=T(
                (2, 12) if self._agent_vs_agent else (12, ),
                {
                    'min': [float("-inf") for _ in range(12)],
                    'max': [float("inf") for _ in range(12)],
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            # 6 valid actions:
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 6,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -5.0,
                    'max': 5.0,
                    'dtype': np.float32,
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self):
        return "DI-engine Slime Volley Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @staticmethod
    def _process_action(action: np.ndarray, _type: str = "binary") -> np.ndarray:
        if action is None:
            return None
        action = action.item()
        # Env receives action in [0, 5] (int type). Can translater into:
        # 1) "binary" type: np.array([0, 1, 0])
        # 2) "atari" type: NOOP, LEFT, UPLEFT, UP, UPRIGHT, RIGHT
        to_atari_action = {
            0: 0,  # NOOP
            1: 4,  # LEFT
            2: 7,  # UPLEFT
            3: 2,  # UP
            4: 6,  # UPRIGHT
            5: 3,  # RIGHT
        }
        to_binary_action = {
            0: [0, 0, 0],  # NOOP
            1: [1, 0, 0],  # LEFT (forward)
            2: [1, 0, 1],  # UPLEFT (forward jump)
            3: [0, 0, 1],  # UP (jump)
            4: [0, 1, 1],  # UPRIGHT (backward jump)
            5: [0, 1, 0],  # RIGHT (backward)
        }
        if _type == "binary":
            return to_ndarray(to_binary_action[action])
        elif _type == "atari":
            return to_atari_action[action]
        else:
            raise NotImplementedError
