from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
import torch
from nervex.utils import build_logger, EasyTimer
from nervex.envs import BaseEnvManager
from .base_serial_collector import CachePool


class BaseSerialEvaluator(object):
    """
    Overview:
        Base class for serial evaluator.
    Interfaces:
        __init__, reset, close, eval
    Property:
        env, policy
    """

    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None  # noqa
    ) -> None:
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
        if env is not None:
            self.env = env
        if policy is not None:
            self.policy = policy
        if tb_logger is not None:
            self._logger, _ = build_logger(path='./log/evaluator', name='evaluator', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(path='./log/evaluator', name='evaluator')
        self._timer = EasyTimer()
        self._cfg = cfg
        self._max_eval_reward = float("-inf")
        self._end_flag = False
        self._last_eval_iter = 0

    @property
    def env(self) -> BaseEnvManager:
        return self._env_manager

    @env.setter
    def env(self, _env_manager: BaseEnvManager) -> None:
        self._end_flag = False
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
        if self._end_flag:
            return
        self._end_flag = True
        self._env_manager.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self):
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        print('train_iter', train_iter, (train_iter - self._last_eval_iter) < self._cfg.eval_freq)
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
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
                obs = self._env_manager.ready_obs
                self._obs_pool.update(obs)
                policy_output = self._policy.forward(obs)
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
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[nerveX serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        return stop_flag, eval_reward
