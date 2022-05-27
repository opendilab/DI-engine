from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
from easydict import EasyDict
from ding.policy import Policy, get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext

import torch
from collections import namedtuple
from ding.utils import EasyTimer, dicts_to_lists
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import CachePool, TrajBuffer, INF, to_tensor_transitions


class BattleCollector():
    def __init__(
            self, 
            cfg: EasyDict, 
            env: BaseEnvManager = None, 
            policy: List[namedtuple] = None
    ):
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False
        self._traj_len = float("inf")

        # TODO(zms) call self.reset() to reset policy and env
        self._policy = policy
        self._env = env 
    
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
            # self._logger.debug(
            #     'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
            #         self._default_n_episode, self._env_num, self._traj_len
            #     )
            # )
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

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self.close()

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  return_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        
        if ctx.n_episode is None:
            ### TODO(zms): self._default_n_episode comes from self.reset_policy()
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                ctx.n_episode = self._default_n_episode
        ### TODO(zms): self._env_num comes from self.reset_env()
        assert ctx.n_episode >= self._env_num, "Please make sure n_episode >= env_num"

        if ctx.policy_kwargs is None:
            ctx.policy_kwargs = {}
        
        collected_episode = 0
        return_data = [[] for _ in range(2)]
        return_info = [[] for _ in range(2)]
        ready_env_id = set()
        remain_episode = ctx.n_episode

        while True:
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)
                obs = {env_id: obs[env_id] for env_id in ready_env_id}

                # Policy forward.
                ### TODO(zms): self._obs_pool comes from self.reset
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i], **ctx.policy_kwargs) for i, p in enumerate(self._policy)]

                ### TODO(zms): self._policy_output_pool comes from self.reset
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {}
                for env_id in ready_env_id:
                    actions[env_id] = []
                    for output in policy_output:
                        actions[env_id].append(output[env_id]['action'])
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                # TODO(zms): self._env_info comes from self.reset()
                self._env_info[env_id]['step'] += 1
                # TODO(zms): self._total_envstep_count comes from self.reset()
                self._total_envstep_count += 1
                with self._timer:
                    for policy_id, policy in enumerate(self._policy):
                        policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                        policy_timestep = type(timestep)(*policy_timestep_data)
                        transition = self._policy[policy_id].process_transition(
                            # TODO(zms): self._policy_output_pool comes from self.reset()
                            self._obs_pool[env_id][policy_id], self._policy_output_pool[env_id][policy_id],
                            policy_timestep
                        )
                        transition['collect_iter'] = ctx.train_iter
                        # TODO(zms): self._traj_buffer comes from self.reset()
                        self._traj_buffer[env_id][policy_id].append(transition)
                        # prepare data
                        if timestep.done:
                            # TODO(zms): to get rid of collector, "to_tensor_transitions" function should be removed and must be somewhere else.
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
                    # TODO(zms): self._total_episode_count comes from self.reset()
                    self._total_episode_count += 1
                    info = {
                        'reward0': timestep.info[0]['final_eval_reward'],
                        'reward1': timestep.info[1]['final_eval_reward'],
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    # TODO(zms): self._episode_info comes from self.reset()
                    self._episode_info.append(info)
                    for i, p in enumerate(self._policy):
                        p.reset([env_id])
                    # TODO(zms): define self._reset_stat 
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
                    for policy_id in range(2):
                        return_info[policy_id].append(timestep.info[policy_id])
            if collected_episode >= ctx.n_episode:
                break
        # log
        ### TODO: how to deal with log here?
        # self._output_log(ctx.train_iter)
        ctx.return_data = return_data
        ctx.return_info = return_info


class StepCollector:
    """
    Overview:
        The class of the collector running by steps, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing \
                the target number of steps.
        Input of ctx:
            - env_step (:obj:`int`): The env steps which will increase during collection.
        """
        old = ctx.env_step
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            # compatible with old config, a train sample = unroll_len step
            target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_step - old >= target_size:
                ctx.trajectories, ctx.trajectory_end_idx = self._transitions.to_trajectories()
                self._transitions.clear()
                break


class EpisodeCollector:
    """
    Overview:
        The class of the collector running by episodes, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing the \
                target number of episodes.
        Input of ctx:
            - env_episode (:obj:`int`): The env env_episode which will increase during collection.
        """
        old = ctx.env_episode
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            target_size = self.cfg.policy.collect.n_episode
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_episode - old >= target_size:
                ctx.episodes = self._transitions.to_episodes()
                self._transitions.clear()
                break


# TODO battle collector
