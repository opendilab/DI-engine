from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
import torch
from nervex.utils import build_logger, EasyTimer, TensorBoardLogger
from .env_manager import BaseEnvManager
from .base_serial_actor import CachePool


class BaseSerialEvaluator(object):

    def __init__(self, cfg: dict) -> None:
        self._default_n_episode = cfg.get('n_episode', None)
        self._stop_val = cfg.stop_val
        self._logger, _ = build_logger(path='./log/evaluator', name='evaluator')
        self._tb_logger = TensorBoardLogger(path='./log/evaluator', name='evaluator')
        for var in ['episode_count', 'step_count', 'avg_step_per_episode', 'avg_time_per_step', 'avg_time_per_episode',
                    'reward_mean', 'reward_std']:
            self._tb_logger.register_var('evaluator/' + var)
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
        self._policy_output_pool = CachePool('policy_output', self._env_num)

    def close(self) -> None:
        self._tb_logger.close()
        self._env.close()

    def eval(self, train_iter: int, n_episode: Optional[int] = None) -> Tuple[bool, float]:
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        episode_count = 0
        step_count = 0
        episode_reward = []
        info = {}
        self.reset()
        self._policy.reset()
        with self._timer:
            while episode_count < n_episode:
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                env_id, obs = self._policy.data_preprocess(obs)
                policy_output = self._policy.forward(env_id, obs)
                policy_output = self._policy.data_postprocess(env_id, policy_output)
                self._policy_output_pool.update(policy_output)
                action = {i: a['action'] for i, a in policy_output.items()}
                timestep = self._env.step(action)
                for i, t in timestep.items():
                    if t.info.get('abnormal', False):
                        # if there is a abnormal timestep, reset all the related variable, also this env has been reset
                        self._policy.reset([i])
                        continue
                    if t.done:
                        # env reset is done by env_manager automatically
                        self._policy.reset([i])
                        reward = t.info['final_eval_reward']
                        if isinstance(reward, torch.Tensor):
                            reward = reward.item()
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
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': episode_count,
            'step_count': step_count,
            'avg_step_per_episode': step_count / episode_count,
            'avg_time_per_step': duration / step_count,
            'avg_time_per_episode': duration / episode_count,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward)
        }
        self._logger.info(
            "[EVALUATOR]evaluate end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )
        tb_vars = [['evaluator/' + k, v, train_iter] for k, v in info.items() if k not in ['train_iter', 'ckpt_name']]
        self._tb_logger.add_val_list(tb_vars, viz_type='scalar')
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= self._stop_val
        if stop_flag:
            self._logger.info(
                "[EVALUATOR] current eval_reward: {} is greater than the stop_val: {}, so the training program is over."
                .format(eval_reward, self._stop_val)
            )
        return stop_flag, eval_reward
