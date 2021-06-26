from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple, deque
from easydict import EasyDict
from functools import reduce
import copy
import numpy as np
import torch

from nervex.utils import build_logger, EasyTimer, deep_merge_dicts, lists_to_dicts
from nervex.envs import BaseEnvManager
from nervex.torch_utils import to_tensor, to_ndarray, tensor_to_list
from .base_serial_collector import CachePool


class BaseSerialEvaluator(object):
    """
    Overview:
        Base class for serial evaluator.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=50,
    )

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
        self._cfg = cfg
        if tb_logger is not None:
            self._logger, _ = build_logger(path='./log/evaluator', name='evaluator', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(path='./log/evaluator', name='evaluator')
        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = cfg.n_episode
        self._stop_value = cfg.stop_value

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = 0
        self._end_flag = False

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self):
        self.close()

    def should_eval(self, train_iter: int) -> bool:
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
        envstep_count = 0
        info = {}
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, reward, eval_monitor.get_current_episode()
                            )
                        )
                    envstep_count += 1
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
            # 'each_reward': episode_reward,
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.print_vars_hor(info)
        # self._logger.print_vars(info)
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
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


class VectorEvalMonitor(object):

    def __init__(self, env_num: int, n_episode: int) -> None:
        assert n_episode >= env_num, "n_episode < env_num, please decrease the number of eval env"
        self._env_num = env_num
        self._n_episode = n_episode
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        for i in range(n_episode % env_num):
            each_env_episode[i] += 1
        self._reward = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}
        self._info = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}

    def is_finished(self) -> bool:
        return all([len(v) == v.maxlen for v in self._reward.values()])

    def update_info(self, env_id: int, info: Any) -> None:
        info = tensor_to_list(info)
        self._info[env_id].append(info)

    def update_reward(self, env_id: int, reward: Any) -> None:
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self._reward[env_id].append(reward)

    def get_episode_reward(self) -> list:
        return sum([list(v) for v in self._reward.values()], [])  # sum(iterable, start)

    def get_current_episode(self) -> int:
        return sum([len(v) for v in self._reward.values()])

    def get_episode_info(self) -> dict:
        if len(self._info[0]) == 0:
            return None
        else:
            total_info = sum([list(v) for v in self._info.values()], [])
            total_info = lists_to_dicts(total_info)
            new_dict = {}
            for k in total_info.keys():
                if np.isscalar(total_info[k][0]):
                    new_dict[k + '_mean'] = np.mean(total_info[k])
            total_info.update(new_dict)
            return total_info
