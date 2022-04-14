from typing import Optional, Any, List, Tuple
from collections import namedtuple, deque
from easydict import EasyDict
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, dicts_to_lists
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, \
    to_tensor_transitions


@SERIAL_COLLECTOR_REGISTRY.register('league_demo')
class LeagueDemoCollector(ISerialCollector):
    """
    Overview:
        League demo collector, derived from BattleEpisodeSerialCollector, add action probs viz.
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    config = dict(deepcopy_obs=False, transform_obs=False, collect_print_freq=100, get_train_sample=False)

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
            assert len(_policy) == 2, "1v1 episode collector needs 2 policy, but found {}".format(len(_policy))
            self._policy = _policy
            self._default_n_episode = _policy[0].get_attribute('cfg').collect.get('n_episode', None)
            self._unroll_len = _policy[0].get_attribute('unroll_len')
            self._on_policy = _policy[0].get_attribute('cfg').on_policy
            self._traj_len = INF
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
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
                     for policy_id in range(2)}
            for env_id in range(self._env_num)
        }
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
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
        self._env_info[env_id] = {'time': 0., 'step': 0}

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

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> Tuple[List[Any], List[Any]]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Returns:
            - return_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
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
        return_data = [[] for _ in range(2)]
        return_info = [[] for _ in range(2)]
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
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i], **policy_kwargs) for i, p in enumerate(self._policy)]
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {}
                for env_id in ready_env_id:
                    actions[env_id] = []
                    for output in policy_output:
                        actions[env_id].append(output[env_id]['action'])
                actions = to_ndarray(actions)
                # temporally for viz
                probs0 = torch.softmax(torch.stack([o['logit'] for o in policy_output[0].values()], 0), 1).mean(0)
                probs1 = torch.softmax(torch.stack([o['logit'] for o in policy_output[1].values()], 0), 1).mean(0)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                self._env_info[env_id]['step'] += 1
                self._total_envstep_count += 1
                with self._timer:
                    for policy_id, policy in enumerate(self._policy):
                        policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                        policy_timestep = type(timestep)(*policy_timestep_data)
                        transition = self._policy[policy_id].process_transition(
                            self._obs_pool[env_id][policy_id], self._policy_output_pool[env_id][policy_id],
                            policy_timestep
                        )
                        transition['collect_iter'] = train_iter
                        self._traj_buffer[env_id][policy_id].append(transition)
                        # prepare data
                        if timestep.done:
                            transitions = to_tensor_transitions(self._traj_buffer[env_id][policy_id])
                            if self._cfg.get_train_sample:
                                train_sample = self._policy[policy_id].get_train_sample(transitions)
                                return_data[policy_id].extend(train_sample)
                            else:
                                return_data[policy_id].append(transitions)
                            self._traj_buffer[env_id][policy_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self._total_episode_count += 1
                    info = {
                        'reward0': timestep.info[0]['final_eval_reward'],
                        'reward1': timestep.info[1]['final_eval_reward'],
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'probs0': probs0,
                        'probs1': probs1,
                    }
                    collected_episode += 1
                    self._episode_info.append(info)
                    for i, p in enumerate(self._policy):
                        p.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
                    for policy_id in range(2):
                        return_info[policy_id].append(timestep.info[policy_id])
            if collected_episode >= n_episode:
                break
        # log
        self._output_log(train_iter)
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
            episode_reward0 = [d['reward0'] for d in self._episode_info]
            episode_reward1 = [d['reward1'] for d in self._episode_info]
            probs0 = [d['probs0'] for d in self._episode_info]
            probs1 = [d['probs1'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward0_mean': np.mean(episode_reward0),
                'reward0_std': np.std(episode_reward0),
                'reward0_max': np.max(episode_reward0),
                'reward0_min': np.min(episode_reward0),
                'reward1_mean': np.mean(episode_reward1),
                'reward1_std': np.std(episode_reward1),
                'reward1_max': np.max(episode_reward1),
                'reward1_min': np.min(episode_reward1),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            info.update(
                {
                    'probs0_select_action0': sum([p[0] for p in probs0]) / len(probs0),
                    'probs0_select_action1': sum([p[1] for p in probs0]) / len(probs0),
                    'probs1_select_action0': sum([p[0] for p in probs1]) / len(probs1),
                    'probs1_select_action1': sum([p[1] for p in probs1]) / len(probs1),
                }
            )
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)
