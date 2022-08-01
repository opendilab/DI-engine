from typing import Optional, Any, List
from collections import namedtuple
from easydict import EasyDict
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, one_time_warning
from ding.torch_utils import to_tensor, to_ndarray
from .base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions


@SERIAL_COLLECTOR_REGISTRY.register('sample')
class SampleSerialCollector(ISerialCollector):
    """
    Overview:
        Sample collector(n_sample), a sample is one training sample for updating model,
        it is usually like <s, a, s', r, d>(one transition)
        while is a trajectory with many transitions, which is often used in RNN-model.
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
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
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

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
            self._default_n_sample = _policy.get_attribute('cfg').collect.get('n_sample', None)
            self._traj_len_inf = _policy.get_attribute('cfg').collect.get('traj_len_inf', False)
            self._unroll_len = _policy.get_attribute('unroll_len')
            self._on_policy = _policy.get_attribute('on_policy')
            if self._default_n_sample is not None and not self._traj_len_inf:
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
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: TrajBuffer}, is used to store traj_len pieces of transitions
        maxlen = self._traj_len if self._traj_len != INF else None
        self._traj_buffer = {env_id: TrajBuffer(maxlen=maxlen) for env_id in range(self._env_num)}
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
        self._traj_buffer[env_id].clear()
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
            policy_kwargs: Optional[dict] = None,
            level_seeds: Optional[List] = None,
    ) -> List[Any]:
        """
        Overview:
            Collect `n_sample` data with policy_kwargs, which is already trained `train_iter` iterations.
        Arguments:
            - n_sample (:obj:`int`): The number of collecting data sample.
            - train_iter (:obj:`int`): The number of training iteration when calling collect method.
            - drop_extra (:obj:`bool`): Whether to drop extra return_data more than `n_sample`.
            - policy_kwargs (:obj:`dict`): The keyword args for policy forward.
            - level_seeds (:obj:`dict`): Used in PLR, represents the seed of the environment that \
                generate the data
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
        collected_sample = 0
        return_data = []

        while collected_sample < n_sample:
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
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
                        # suppose there is no reset param, just reset this env
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('Env{} returns a abnormal step, its info is {}'.format(env_id, timestep.info))
                        continue
                    if 'type' in self._policy.get_attribute('cfg') and \
                            self._policy.get_attribute('cfg').type == 'ngu_command':
                        # for NGU policy
                        transition = self._policy.process_transition(
                            self._obs_pool[env_id], self._policy_output_pool[env_id], timestep, env_id
                        )
                    else:
                        transition = self._policy.process_transition(
                            self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                        )
                        if level_seeds is not None:
                            transition['seed'] = level_seeds[env_id]
                    # ``train_iter`` passed in from ``serial_entry``, indicates current collecting model's iteration.
                    transition['collect_iter'] = train_iter
                    self._traj_buffer[env_id].append(transition)
                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1
                    # prepare data
                    if timestep.done or len(self._traj_buffer[env_id]) == self._traj_len:
                        # If policy is r2d2:
                        # 1. For each collect_env, we want to collect data of length self._traj_len=INF
                        # unless the episode enters the 'done' state.
                        # 2. The length of a train (sequence) sample in r2d2 is <burnin + learn_unroll_length>
                        # (please refer to r2d2.py) and in each collect phase,
                        # we collect a total of <n_sample> (sequence) samples.
                        # 3. When timestep is done and we only collected very few transitions in self._traj_buffer,
                        # by going through self._policy.get_train_sample, it will be padded automatically to get the
                        # sequence sample of length <burnin + learn_unroll_len> (please refer to r2d2.py).

                        # Episode is done or traj_buffer(maxlen=traj_len) is full.
                        transitions = to_tensor_transitions(self._traj_buffer[env_id])
                        train_sample = self._policy.get_train_sample(transitions)
                        return_data.extend(train_sample)
                        self._total_train_sample_count += len(train_sample)
                        self._env_info[env_id]['train_sample'] += len(train_sample)
                        collected_sample += len(train_sample)
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
                        'train_sample': self._env_info[env_id]['train_sample'],
                    }
                    self._episode_info.append(info)
                    # Env reset is done by env_manager automatically
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
        # log
        self._output_log(train_iter)
        # on-policy reset
        if self._on_policy:
            for env_id in range(self._env_num):
                self._reset_stat(env_id)

        if drop_extra:
            return return_data[:n_sample]
        else:
            return return_data

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
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_train_sample_count': self._total_train_sample_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
                # 'each_reward': episode_reward,
            }
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)
