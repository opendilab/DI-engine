from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
import torch
from nervex.utils import build_logger, EasyTimer
from nervex.envs import BaseEnvManager
from .base_serial_actor import CachePool


class BaseSerialEvaluator(object):
    """
    Overview:
        Base class for serial evaluator.
    Interfaces:
        __init__, reset, close, eval
    Property:
        env, policy
    """

    def __init__(self, cfg: dict, tb_logger: 'SummaryWriter' = None) -> None:  # noqa
        """
        Overview:
            Init method. Load config and use ``self._cfg`` setting to build common serial evaluator components,
            e.g. logger helper, timer.
            Policy is not initialized here, but set afterwards through policy setter.
        Arguments:
            - cfg (:obj:`EasyDict`)
        """
        self._default_n_episode = cfg.get('n_episode', None)
        self._stop_value = cfg.stop_value
        if tb_logger is not None:
            self._logger, _ = build_logger(path='./log/evaluator', name='evaluator', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(path='./log/evaluator', name='evaluator')
        self._timer = EasyTimer()
        self._cfg = cfg

    @property
    def env(self) -> BaseEnvManager:
        return self._env_manager

    @env.setter
    def env(self, _env_manager: BaseEnvManager) -> None:
        self._env_manager = _env_manager
        self._env_manager.launch()
        self._env_num = self._env_manager.env_num

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
        self._env_manager.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def eval(self, train_iter: int = -1, envstep: int = -1, n_episode: Optional[int] = None) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy.
        Arguments:
            - train_iter (:obj:`int`): Current training iteration.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        episode_count = 0
        envstep_count = 0
        episode_reward = []
        info = {}
        self.reset()
        self._policy.reset()
        with self._timer:
            while episode_count < n_episode:
                obs = self._env_manager.next_obs
                self._obs_pool.update(obs)
                env_id, obs = self._policy.data_preprocess(obs)
                policy_output = self._policy.forward(env_id, obs)
                policy_output = self._policy.data_postprocess(env_id, policy_output)
                self._policy_output_pool.update(policy_output)
                actions = {i: a['action'] for i, a in policy_output.items()}
                timesteps = self._env_manager.step(actions)
                for i, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([i])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
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
                    envstep_count += 1
        duration = self._timer.value
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': episode_count,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / episode_count,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': episode_count / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'each_reward': episode_reward,
        }
        # self._logger.print_vars(info)
        self._logger.info(
            "[EVALUATOR] Evaluation ends:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            self._tb_logger.add_scalar('evaluator_iter/' + k, v, train_iter)
            self._tb_logger.add_scalar('evaluator_step/' + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= self._stop_value
        if stop_flag:
            self._logger.info(
                "[EVALUATOR] Current eval_reward: {} is greater than stop_value: {}, so the training program is over.".
                format(eval_reward, self._stop_value)
            )
        return stop_flag, eval_reward
