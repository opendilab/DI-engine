from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from nervex.utils import build_logger, EasyTimer
from .env_manager import BaseEnvManager
from .base_serial_actor import CachePool


class BaseSerialEvaluator(object):

    def __init__(self, cfg: dict) -> None:
        self._default_n_episode = cfg.get('n_episode', None)
        self._stop_val = cfg.stop_val
        self._logger = build_logger()
        self._timer = EasyTimer()
        self._cfg = cfg

    @property
    def env(self) -> BaseEnvManager:
        return self._env

    @env.setter
    def env(self, _env: BaseEnvManager) -> None:
        self._env = _env
        self._env.launch()
        self._env_num = self._env.env_num

    @property
    def policy(self) -> namedtuple:
        return self._policy

    @policy.setter
    def policy(self, _policy: namedtuple) -> None:
        self._policy = _policy

    def reset(self) -> None:
        self._obs_pool = CachePool('obs', self._env_num)
        self._agent_output_pool = CachePool('agent_output', self._env_num)
        self._env.reset()

    def eval(self, n_episode: Optional[int] = None) -> bool:
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        episode_count = 0
        step_count = 0
        episode_reward = {}
        info = {}
        self.reset()
        with self._timer:
            while episode_count < n_episode:
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                obs, env_id = self._policy.data_preprocess(obs)
                agent_output = self._policy.forward(obs)
                agent_output = self._policy.data_postprocess(env_id, agent_output)
                self._agent_output_pool.update(agent_output)
                action = {i: a['action'] for i, a in agent_output.items()}
                timestep = self._env.step(action)
                for i, t in timestep.items():
                    if t.done:
                        # env reset is done by env_manager automatically
                        self._policy.callback_episode_done(i)
                        reward = t.info[i]['final_eval_reward']
                        episode_reward.append(reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                i, reward, episode_count
                            )
                        )
                        episode_count += 1
                    step_count += 1
        duration = self._timer.value
        info = {
            'episode_count': episode_count,
            'step_count': step_count,
            'avg_step_per_episode': step_count / episode_count,
            'avg_time_per_step': duration / step_count,
            'avg_time_per_episode': duration / episode_count,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward)
        }
        self._logger.info(
            "[EVALUATOR]collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )
        return np.mean(episode_reward) >= self._stop_val
