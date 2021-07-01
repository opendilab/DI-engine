from typing import Optional, Any, List
from collections import namedtuple, deque
from easydict import EasyDict
import numpy as np
import torch

from nervex.envs import BaseEnvManager
from nervex.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from nervex.torch_utils import to_tensor, to_ndarray
from .base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions


@SERIAL_COLLECTOR_REGISTRY.register('episode')
class EpisodeCollector(ISerialCollector):
    """
    Overview:
        Episode collector(n_episode)
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    config = dict(deepcopy_obs=False, transform_obs=False, collect_print_freq=100)

    def __init__(
            self,
            cfg: EasyDict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None  # noqa
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
        """
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False

        if tb_logger is not None:
            self._logger, _ = build_logger(path='./log/collector', name='collector', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(path='./log/collector', name='collector')
        self.reset(policy, env)

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
            self._default_n_episode = _policy.get_attribute('cfg').collect.get('n_episode', None)
            self._unroll_len = _policy.get_attribute('unroll_len')
            self._on_policy = _policy.get_attribute('cfg').on_policy
            self._traj_len = INF
            self._logger.info(
                'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: TrajBuffer}, is used to store traj_len pieces of transitions
        self._traj_buffer = {env_id: TrajBuffer(maxlen=self._traj_len) for env_id in range(self._env_num)}
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False

    def _reset_stat(self, env_id: int) -> None:
        self._traj_buffer[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0}

    @property
    def envstep(self) -> int:
        return self._total_envstep_count

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        self.close()

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Returns:
            - return_episode (:obj:`List`): A list containing collected episodes.
        """
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num"
        if policy_kwargs is None:
            policy_kwargs = {}
        collected_episode = 0
        return_episode = []
        ready_env_id = set()
        remain_episode = n_episode

        while True:
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)
                obs = {env_id: obs[env_id] for env_id in ready_env_id}
                # Policy forward.
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs, **policy_kwargs)
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                with self._timer:
                    if timestep.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._env.reset([env_id])
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('env_id abnormal step', env_id, timestep.info)
                        continue
                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    # ``train_iter`` passed in from ``serial_entry``, indicates current collecting model's iteration.
                    transition['collect_iter'] = train_iter
                    self._traj_buffer[env_id].append(transition)
                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1
                    # prepare data
                    if timestep.done:
                        transitions = to_tensor_transitions(self._traj_buffer[env_id])
                        return_episode.append(transitions)
                        self._traj_buffer[env_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self._total_episode_count += 1
                    reward = timestep.info['final_eval_reward']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
            if collected_episode >= n_episode:
                break
        # log
        self._output_log(train_iter)
        return return_episode

    def _output_log(self, train_iter: int) -> None:
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
                # 'each_reward': episode_reward,
            }
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            # self._logger.print_vars_hor(info)
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('collector_iter/' + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('collector_step/' + k, v, self._total_envstep_count)
