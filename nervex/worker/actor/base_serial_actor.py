from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple, deque
import copy
import numpy as np
import torch

from .env_manager import BaseEnvManager
from nervex.utils import build_logger, EasyTimer


class BaseSerialActor(object):
    """
    Overview:
        Abstract baseclass for serial actor.
    Interfaces:
        __init__, reset, generate_data, close, _collect_episode, _collect_sample, _collect
    Property:
        env, policy,
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
        """
        self._default_n_episode = cfg.get('n_episode', None)
        self._default_n_sample = cfg.get('n_sample', None)
        self._traj_len = cfg.traj_len
        if self._traj_len == "inf":
            raise ValueError(
                "Serial Actor must indicate finite traj_len, if you want to use the total episode, \
                please set it equal to the maximum length of the env's episode"
            )
        self._traj_cache_length = self._traj_len
        self._traj_print_freq = cfg.traj_print_freq
        self._collect_print_freq = cfg.collect_print_freq
        self._logger, _ = build_logger(path='./log/actor', name='actor')
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
        self.reset()

    @property
    def policy(self) -> namedtuple:
        return self._policy

    @policy.setter
    def policy(self, _policy: namedtuple) -> None:
        self._policy = _policy

    def reset(self) -> None:
        self._obs_pool = CachePool('obs', self._env_num)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_cache = {env_id: deque}, used to store traj_len pieces of transitions
        self._traj_cache = {env_id: deque(maxlen=self._traj_cache_length) for env_id in range(self._env_num)}
        self._total_collect_step = 0
        self._total_step = 0
        self._total_episode = 0
        self._total_sample = 0
        self._total_duration = 0

    def generate_data(self,
                      iter_count: int,
                      n_episode: Optional[int] = None,
                      n_sample: Optional[int] = None) -> Tuple[List[Any], dict]:
        """
       Overview:
           Generate data. ``n_episode`` and ``n_sample`` can't be not None at the same time.
       Arguments:
           - iter_count (:obj:`int`): count of iteration
           - n_episode (:obj:`int`): number of episode
           - n_sample (:obj:`int`): number of sample
       Returns:
           - return_data (:obj:`List`): A list containing training samples.
           - collect_info (:obj:`dict`): A dict containing sample collection information.
       """
        assert n_episode is None or n_sample is None, "n_episode and n_sample can't be not None at the same time"
        if n_episode is not None:
            return self._collect_episode(iter_count, n_episode)
        elif n_sample is not None:
            return self._collect_sample(iter_count, n_sample)
        elif self._default_n_episode is not None:
            return self._collect_episode(iter_count, self._default_n_episode)
        elif self._default_n_sample is not None:
            return self._collect_sample(iter_count, self._default_n_sample)
        else:
            raise RuntimeError("please clarify specific n_episode or n_sample(int value) in config yaml or outer call")

    def close(self) -> None:
        self._env.close()

    def _collect_episode(self, iter_count: int, n_episode: int) -> Tuple[List[Any], dict]:
        return self._collect(iter_count, lambda x, y: x >= n_episode)

    def _collect_sample(self, iter_count: int, n_sample: int) -> Tuple[List[Any], dict]:
        return self._collect(iter_count, lambda x, y: y >= n_sample)

    def _collect(self, iter_count: int, collect_end_fn: Callable) -> Tuple[List[Any], dict]:
        """
        Overview:
            Collect function for generate data. Called by ``self._collect_episode`` and ``self._collect_sample``.
        Arguments:
            - iter_count (:obj:`int`): count of iteration
            - collect_end_fn (:obj:`Callable`): end of collect
        Returns:
            - return_data (:obj:`List`): A list containing training samples.
            - collect_info (:obj:`dict`): A dict containing sample collection information.
        """
        episode_count = 0
        step_count = 0
        train_sample_count = 0
        episode_reward = []
        return_data = []
        info = {}
        self._policy.reset()
        with self._timer:
            while not collect_end_fn(episode_count, train_sample_count):
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                env_id, obs = self._policy.data_preprocess(obs)
                policy_output = self._policy.forward(env_id, obs)
                policy_output = self._policy.data_postprocess(env_id, policy_output)
                self._policy_output_pool.update(policy_output)
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                timesteps = self._env.step(actions)
                for env_id, timestep in timesteps.items():
                    if timestep.info.get('abnormal', False):
                        # if there is a abnormal timestep, reset all the related variable, also this env has been reset
                        self._traj_cache[env_id].clear()
                        self._obs_pool.reset(env_id)
                        self._policy_output_pool.reset(env_id)
                        self._policy.reset([env_id])
                        print('env_id abnormal step', env_id, timestep.info)
                        continue
                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    # parameter ``iter_count``, which is passed in from ``serial_entry``, indicates current
                    # collecting model's iteration
                    transition['collect_iter'] = iter_count
                    self._traj_cache[env_id].append(transition)
                    if timestep.done or len(self._traj_cache[env_id]) == self._traj_len:
                        # episode is done or traj_cache(maxlen=traj_len) is full
                        train_sample = self._policy.get_train_sample(self._traj_cache[env_id])
                        return_data.extend(train_sample)
                        train_sample_count += len(train_sample)
                        self._total_sample += len(train_sample)
                        # if (train_sample_count + 1) % self._traj_print_freq == 0:
                        #     self._logger.info(
                        #         "env {} get new traj, collected traj: {}".format(env_id, train_sample_count)
                        #     )
                    if timestep.done:
                        # env reset is done by env_manager automatically
                        self._traj_cache[env_id].clear()
                        self._obs_pool.reset(env_id)
                        self._policy_output_pool.reset(env_id)
                        self._policy.reset([env_id])
                        reward = timestep.info['final_eval_reward']
                        if isinstance(reward, torch.Tensor):
                            reward = reward.item()
                        episode_reward.append(reward)
                        # self._logger.info(
                        #     "env {} finish episode, final reward: {}, collected episode: {}".format(
                        #         env_id, reward, episode_count
                        #     )
                        # )
                        episode_count += 1
                        self._total_episode += 1
                    step_count += 1
                    self._total_step += 1
        duration = self._timer.value
        if (self._total_collect_step + 1) % self._collect_print_freq == 0:
            info = {
                'episode_count': episode_count,
                'step_count': step_count,
                'train_sample_count': train_sample_count,
                'avg_step_per_episode': step_count / max(1, episode_count),
                'avg_sample_per_epsiode': train_sample_count / max(1, episode_count),
                'avg_time_per_step': duration / (step_count + 1e-8),
                'avg_time_per_train_sample': duration / (train_sample_count + 1e-8),
                'avg_time_per_episode': duration / max(1, episode_count),
                'reward_mean': np.mean(episode_reward) if len(episode_reward) > 0 else 0.,
                'reward_std': np.std(episode_reward) if len(episode_reward) > 0 else 0.,
                'each_reward': episode_reward,
            }
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
        self._total_collect_step += 1
        self._total_duration += duration
        collect_info = {
            'total_collect_step': self._total_collect_step,
            'total_step': self._total_step,
            'total_sample': self._total_sample,
            'total_episode': self._total_episode,
            'total_duration': self._total_duration,
        }
        return return_data, collect_info


class CachePool(object):
    """
    Overview:
       CachePool is the repository of cache items.
    Interfaces:
        __init__, update, __getitem__, reset
    """

    def __init__(self, name: str, env_num: int, deepcopy: bool = False):
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

    def update(self, data: Dict[int, Any]):
        for i, d in data.items():
            if self._deepcopy:
                self._pool[i] = copy.deepcopy(d)
            else:
                self._pool[i] = d

    def __getitem__(self, idx: int) -> Any:
        return self._pool[idx]

    def reset(self, idx: int) -> None:
        self._pool[idx] = None
