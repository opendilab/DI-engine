from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from collections import namedtuple, deque
from easydict import EasyDict
import copy
import numpy as np
import torch

from nervex.envs import BaseEnvManager
from nervex.utils import build_logger, EasyTimer
from nervex.torch_utils import to_tensor, to_ndarray

INF = float("inf")


class BaseSerialCollector(object):
    """
    Overview:
        Baseclass for serial collector.
    Interfaces:
        __init__, reset, collect_data, close
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(collect_print_freq=100, )

    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None  # noqa
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
        """
        self._collect_print_freq = cfg.collect_print_freq
        if tb_logger is not None:
            self._logger, _ = build_logger(path='./log/collector', name='collector', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(path='./log/collector', name='collector')
        if env is not None:
            self.env = env
        if policy is not None:
            self.policy = policy
        self._timer = EasyTimer()
        self._cfg = cfg
        self._end_flag = False

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
        assert hasattr(self, '_env_manager'), "please set env first"
        self._policy = _policy
        self._default_n_episode = _policy.get_attribute('cfg').collect.get('n_episode', None)
        self._default_n_sample = _policy.get_attribute('cfg').collect.get('n_sample', None)
        self._unroll_len = _policy.get_attribute('unroll_len')
        assert any(
            [t is None for t in [self._default_n_sample, self._default_n_episode]]
        ), "n_episode/n_sample in policy cfg can't be not None at the same time"
        if self._default_n_episode is not None:
            self._traj_len = INF
            self._logger.info(
                'Set default n_episode mode(n_sample({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
        elif self._default_n_sample is not None:
            self._traj_len = max(
                self._unroll_len,
                self._default_n_sample // self._env_num + int(self._default_n_sample % self._env_num != 0)
            )
            self._logger.info(
                'Set default n_sample mode(n_sample({}), env_num({}), traj_len({}))'.format(
                    self._default_n_sample, self._env_num, self._traj_len
                )
            )
        else:
            self._traj_len = INF
        self.reset()

    def reset(self) -> None:
        self._obs_pool = CachePool('obs', self._env_num)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_cache is {env_id: deque}, is used to store traj_len pieces of transitions
        traj_len = self._traj_len if self._traj_len != INF else None
        self._traj_cache = {env_id: deque(maxlen=traj_len) for env_id in range(self._env_num)}
        self._env_info = {env_id: {'time': 0., 'step': 0, 'train_sample': 0} for env_id in range(self._env_num)}
        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_train_sample_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._done_episode = []

    @property
    def collector_info(self) -> dict:
        """
        Overview:
            Get current info dict, which will be sent to commander, e.g. replay buffer priority update,
            current iteration, hyper-parameter adjustment, whether task is finished, etc.
        Returns:
            - info (:obj:`dict`): Current learner info dict.
        """
        ret = {'env_step': self._total_envstep_count, 'sample_step': self._total_train_sample_count}
        return ret

    def collect_data(
            self,
            train_iter: int = -1,
            n_episode: Optional[int] = None,
            n_sample: Optional[int] = None,
            policy_kwargs: Optional[dict] = None
    ) -> Tuple[List[Any], dict]:
        """
        Overview:
           Collect data. Either ``n_episode`` or ``n_sample`` must be None.
        Arguments:
           - train_iter (:obj:`int`): count of iteration
           - n_episode (:obj:`int`): number of episode
           - n_sample (:obj:`int`): number of sample
        Returns:
           - return_data (:obj:`List`): A list containing training samples.
        """
        assert n_episode is None or n_sample is None, "Either n_episode or n_sample must be None"
        if n_episode is not None:
            return self._collect_episode(train_iter, n_episode, policy_kwargs)
        elif n_sample is not None:
            return self._collect_sample(train_iter, n_sample, policy_kwargs)
        elif self._default_n_episode is not None:
            return self._collect_episode(train_iter, self._default_n_episode, policy_kwargs)
        elif self._default_n_sample is not None:
            return self._collect_sample(train_iter, self._default_n_sample, policy_kwargs)
        else:
            raise RuntimeError("Please clarify specific n_episode or n_sample (int value) in config yaml or outer call")

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        self._env_manager.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        self.close()

    def _collect_episode(self,
                         train_iter: int,
                         n_episode: int,
                         policy_kwargs: Optional[dict] = None) -> Tuple[List[Any], dict]:
        return self._collect(train_iter, lambda num_episode, num_sample: num_episode >= n_episode, policy_kwargs)

    def _collect_sample(self,
                        train_iter: int,
                        n_sample: int,
                        policy_kwargs: Optional[dict] = None) -> Tuple[List[Any], dict]:
        return self._collect(train_iter, lambda num_episode, num_sample: num_sample >= n_sample, policy_kwargs)

    def _collect(self,
                 train_iter: int,
                 collect_end_fn: Callable,
                 policy_kwargs: Optional[dict] = None) -> Tuple[List[Any], dict]:
        """
        Overview:
            Real collect method in process of generating data.
            Called by ``self._collect_episode`` and ``self._collect_sample``.
        Arguments:
            - train_iter (:obj:`int`): count of iteration
            - collect_end_fn (:obj:`Callable`): end of collect
        Returns:
            - return_data (:obj:`List`): A list containing training samples.
        """
        if policy_kwargs is None:
            policy_kwargs = {}
        return_data = []
        self._policy.reset()
        collected_episode = 0
        collected_sample = 0
        while not collect_end_fn(collected_episode, collected_sample):
            with self._timer:
                # Get current env obs.
                obs = self._env_manager.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                self._obs_pool.update(obs)
                # Policy forward.
                policy_output = self._policy.forward(obs, **policy_kwargs)
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env_manager.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
            single_env_pre_duration = self._timer.value / len(timesteps)
            # For each env:
            # 1) Form a transition and store it in cache;
            # 2) Get a train sample from cache if cache is full or env is done;
            # 3) Reset related variables if env is done.
            for env_id, timestep in timesteps.items():
                with self._timer:
                    if timestep.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._var_reset(env_id)
                        self._logger.info('env_id abnormal step', env_id, timestep.info)
                        continue
                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    # ``train_iter`` passed in from ``serial_entry``, indicates current collecting model's iteration.
                    transition['collect_iter'] = train_iter
                    self._traj_cache[env_id].append(transition)
                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1
                    # prepare data
                    if timestep.done or len(self._traj_cache[env_id]) == self._traj_len:
                        # Episode is done or traj_cache(maxlen=traj_len) is full.
                        train_sample = self._policy.get_train_sample(self._traj_cache[env_id])
                        return_data.extend(train_sample)
                        self._total_train_sample_count += len(train_sample)
                        self._env_info[env_id]['train_sample'] += len(train_sample)
                        collected_sample += len(train_sample)
                        self._traj_cache[env_id].clear()
                    # Reset if env is done.
                    if timestep.done:
                        # Env reset is done by env_manager automatically
                        self._var_reset(env_id)
                        self._total_episode_count += 1
                        collected_episode += 1
                self._env_info[env_id]['time'] += self._timer.value + single_env_pre_duration
                # If env is done, record episode info and reset env info
                if timestep.done:
                    reward = timestep.info['final_eval_reward']
                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'train_sample': self._env_info[env_id]['train_sample'],
                    }
                    self._episode_info.append(info)
                    self._env_info[env_id] = {'time': 0., 'step': 0, 'train_sample': 0}
        # log
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            train_sample_count = sum([d['train_sample'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'train_sample_count': train_sample_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_sample_per_episode': train_sample_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_train_sample_per_sec': train_sample_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'each_reward': episode_reward,
                'total_envstep_count': self._total_envstep_count,
                'total_train_sample_count': self._total_train_sample_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            self._episode_info.clear()
            # self._logger.print_vars(info)
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('collector_iter/' + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('collector_step/' + k, v, self._total_envstep_count)
        return return_data

    def _var_reset(self, env_id: int) -> None:
        """
        Overview:
           When an env is to be reset, e.g. episode is done, abnormal occurs, etc.
           Reset corresponding vars, e.g. traj_cache, obs_pool, policy_output_pool, policy.
        Arguments:
           - env_id (:obj:`int`): Id of the env which is to be reset.
        """
        self._traj_cache[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._policy.reset([env_id])

    @property
    def envstep(self) -> int:
        return self._total_envstep_count


class CachePool(object):
    """
    Overview:
       CachePool is the repository of cache items.
    Interfaces:
        __init__, update, __getitem__, reset
    """

    def __init__(self, name: str, env_num: int, deepcopy: bool = False) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - name (:obj:`str`): name of cache
            - env_num (:obj:`int`): number of environments
            - deepcopy (:obj:`bool`): whether to deepcopy data
        """
        self._pool = [None for _ in range(env_num)]
        # TODO(nyz) whether must use deepcopy
        self._deepcopy = deepcopy

    def update(self, data: Union[Dict[int, Any], list]) -> None:
        """
        Overview:
            Update elements in cache pool.
        Arguments:
            - data (:obj:`Dict[int, Any]`): A dict containing update index-value pairs. Key is index in cache pool, \
                and value is the new element.
        """
        if isinstance(data, dict):
            data = [data]
        for index in range(len(data)):
            for i in data[index].keys():
                d = data[index][i]
                if self._deepcopy:
                    copy_d = copy.deepcopy(d)
                else:
                    copy_d = d
                if index == 0:
                    self._pool[i] = [copy_d]
                else:
                    self._pool[i].append(copy_d)

    def __getitem__(self, idx: int) -> Any:
        data = self._pool[idx]
        if len(data) == 1:
            data = data[0]
        return data

    def reset(self, idx: int) -> None:
        self._pool[idx] = None
