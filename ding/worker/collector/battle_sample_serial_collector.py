from typing import Optional, Any, List, Tuple
from collections import namedtuple
from easydict import EasyDict
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, dicts_to_lists, one_time_warning
from ding.torch_utils import to_tensor, to_ndarray
from .base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions


@SERIAL_COLLECTOR_REGISTRY.register('sample_1v1')
class BattleSampleSerialCollector(ISerialCollector):
    """
    Overview:
        Sample collector(n_sample) with multiple(n VS n) policy battle
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
            policy: List[namedtuple] = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector'
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`List[namedtuple]`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False

        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self._traj_len = float("inf")
        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
        Arguments:
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[List[namedtuple]] = None) -> None:
        """
        Overview:
            Reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of collect_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            assert len(_policy) > 1, "battle sample collector needs more than 1 policy, but found {}".format(
                len(_policy)
            )
            self._policy = _policy
            self._policy_num = len(self._policy)
            self._default_n_sample = _policy[0].get_attribute('cfg').collect.get('n_sample', None)
            self._unroll_len = _policy[0].get_attribute('unroll_len')
            self._on_policy = _policy[0].get_attribute('cfg').on_policy
            self._policy_collect_data = [
                getattr(self._policy[i], 'collect_data', True) for i in range(self._policy_num)
            ]
            if self._default_n_sample is not None:
                self._traj_len = max(
                    self._unroll_len,
                    self._default_n_sample // self._env_num + int(self._default_n_sample % self._env_num != 0)
                )
                self._logger.debug(
                    'Set default n_sample mode(n_sample({}), env_num({}), traj_len({}))'.format(
                        self._default_n_sample, self._env_num, self._traj_len
                    )
                )
            else:
                self._traj_len = INF
        for p in self._policy:
            p.reset()

    def reset(self, _policy: Optional[List[namedtuple]] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of collect_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: {policy_id: TrajBuffer}}, is used to store traj_len pieces of transitions
        self._traj_buffer = {
            env_id: {policy_id: TrajBuffer(maxlen=self._traj_len)
                     for policy_id in range(self._policy_num)}
            for env_id in range(self._env_num)
        }
        self._env_info = {env_id: {'time': 0., 'step': 0, 'train_sample': 0} for env_id in range(self._env_num)}

        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_train_sample_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False

    def _reset_stat(self, env_id: int) -> None:
        """
        Overview:
            Reset the collector's state. Including reset the traj_buffer, obs_pool, policy_output_pool\
                and env_info. Reset these states according to env_id. You can refer to base_serial_collector\
                to get more messages.
        Arguments:
            - env_id (:obj:`int`): the id where we need to reset the collector's state
        """
        for i in range(2):
            self._traj_buffer[env_id][i].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0, 'train_sample': 0}

    @property
    def envstep(self) -> int:
        """
        Overview:
            Print the total envstep count.
        Return:
            - envstep (:obj:`int`): the total envstep count
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self.close()

    def collect(
            self,
            n_sample: Optional[int] = None,
            train_iter: int = 0,
            drop_extra: bool = True,
            policy_kwargs: Optional[dict] = None
    ) -> Tuple[List[Any], List[Any]]:
        """
        Overview:
            Collect `n_sample` data with policy_kwargs, which is already trained `train_iter` iterations.
        Arguments:
            - n_sample (:obj:`int`): The number of collecting data sample.
            - train_iter (:obj:`int`): The number of training iteration when calling collect method.
            - drop_extra (:obj:`bool`): Whether to drop extra return_data more than `n_sample`.
            - policy_kwargs (:obj:`dict`): The keyword args for policy forward.
        Returns:
            - return_data (:obj:`List`): A list containing training samples.
        """
        if n_sample is None:
            if self._default_n_sample is None:
                raise RuntimeError("Please specify collect n_sample")
            else:
                n_sample = self._default_n_sample
        if n_sample % self._env_num != 0:
            one_time_warning(
                "Please make sure env_num is divisible by n_sample: {}/{}, ".format(n_sample, self._env_num) +
                "which may cause convergence problems in a few algorithms"
            )
        if policy_kwargs is None:
            policy_kwargs = {}
        collected_sample = [0 for _ in range(self._policy_num)]
        return_data = [[] for _ in range(self._policy_num)]
        return_info = [[] for _ in range(self._policy_num)]

        while any([c < n_sample for i, c in enumerate(collected_sample) if self._policy_collect_data[i]]):
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                # Policy forward.
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i], **policy_kwargs) for i, p in enumerate(self._policy)]
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {}
                for policy_output_item in policy_output:
                    for env_id, output in policy_output_item.items():
                        if env_id not in actions:
                            actions[env_id] = []
                        actions[env_id].append(output['action'])
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                self._env_info[env_id]['step'] += 1
                self._total_envstep_count += 1
                with self._timer:
                    for policy_id, policy in enumerate(self._policy):
                        if not self._policy_collect_data[policy_id]:
                            continue
                        policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                        policy_timestep = type(timestep)(*policy_timestep_data)
                        transition = self._policy[policy_id].process_transition(
                            self._obs_pool[env_id][policy_id], self._policy_output_pool[env_id][policy_id],
                            policy_timestep
                        )
                        transition['collect_iter'] = train_iter
                        self._traj_buffer[env_id][policy_id].append(transition)
                        # prepare data
                        if timestep.done or len(self._traj_buffer[env_id][policy_id]) == self._traj_len:
                            transitions = to_tensor_transitions(self._traj_buffer[env_id][policy_id])
                            train_sample = self._policy[policy_id].get_train_sample(transitions)
                            return_data[policy_id].extend(train_sample)
                            self._total_train_sample_count += len(train_sample)
                            self._env_info[env_id]['train_sample'] += len(train_sample)
                            collected_sample[policy_id] += len(train_sample)
                            self._traj_buffer[env_id][policy_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self._total_episode_count += 1
                    info = {
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'train_sample': self._env_info[env_id]['train_sample'],
                    }
                    for i in range(self._policy_num):
                        info['reward{}'.format(i)] = timestep.info[i]['final_eval_reward']
                    self._episode_info.append(info)
                    for i, p in enumerate(self._policy):
                        p.reset([env_id])
                    self._reset_stat(env_id)
                    for policy_id in range(2):
                        return_info[policy_id].append(timestep.info[policy_id])
        # log
        self._output_log(train_iter)
        return_data = [r[:n_sample] for r in return_data]
        if drop_extra:
            return_data = return_data[:n_sample]
        return return_data, return_info

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Print the output log information. You can refer to Docs/Best Practice/How to understand\
             training generated folders/Serial mode/log/collector for more details.
        Arguments:
            - train_iter (:obj:`int`): the number of training iteration.
        """
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = []
            for i in range(self._policy_num):
                episode_reward_item = [d['reward{}'.format(i)] for d in self._episode_info]
                episode_reward.append(episode_reward_item)
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            for k, fn in {'mean': np.mean, 'std': np.std, 'max': np.max, 'min': np.min}.items():
                for i in range(self._policy_num):
                    # such as reward0_mean
                    info['reward{}_{}'.format(i, k)] = fn(episode_reward[i])
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)
