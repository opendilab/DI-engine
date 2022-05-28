from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
from easydict import EasyDict
from ding.policy import Policy, get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList

# if TYPE_CHECKING:
from ding.framework import OnlineRLContext

import torch
from collections import namedtuple
from ding.utils import EasyTimer, dicts_to_lists
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import CachePool, TrajBuffer, INF, to_tensor_transitions

def reset_policy():

    def _reset_policy(ctx: OnlineRLContext):
        if ctx._policy is not None:
            assert len(ctx._policy) > 1, "battle collector needs more than 1 policies"
            ctx._default_n_episode = ctx._policy[0].get_attribute('cfg').collect.get('n_episode', None)

        for p in ctx._policy:
            p.reset()

    return _reset_policy




class BattleCollector:
    def __init__(
            self, 
            cfg: EasyDict, 
            env: BaseEnvManager, 
    ):
        self.cfg = cfg
        self._timer = EasyTimer()
        self.end_flag = False
        self.traj_len = float("inf")
        # self._reset(env)
        self.env = env
        self.env.launch()
        self.env_num = self.env.env_num

        self.obs_pool = CachePool('obs', self.env_num, deepcopy=self.cfg.deepcopy_obs)
        self.policy_output_pool = CachePool('policy_output', self.env_num)
        # _traj_buffer is {env_id: {policy_id: TrajBuffer}}, is used to store traj_len pieces of transitions
        self.traj_buffer = {
            env_id: {policy_id: TrajBuffer(maxlen=self.traj_len)
                     for policy_id in range(2)}
            for env_id in range(self.env_num)
        }
        self.env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self.env_num)}

        self.episode_info = []
        self.total_envstep_count = 0
        self.total_episode_count = 0
        self.end_flag = False
    
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
            self.traj_buffer[env_id][i].clear()
        self.obs_pool.reset(env_id)
        self.policy_output_pool.reset(env_id)
        self.env_info[env_id] = {'time': 0., 'step': 0}
    
    def _close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self._close()

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
        ctx.envstep = self.total_envstep_count
        if ctx.n_episode is None:
            if ctx._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                ctx.n_episode = ctx._default_n_episode
        assert ctx.n_episode >= self.env_num, "Please make sure n_episode >= env_num"

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
                obs = self.env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)
                obs = {env_id: obs[env_id] for env_id in ready_env_id}

                # Policy forward.
                self.obs_pool.update(obs)
                if self.cfg.transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i], **ctx.policy_kwargs) for i, p in enumerate(ctx._policy)]

                self.policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {}
                for env_id in ready_env_id:
                    actions[env_id] = []
                    for output in policy_output:
                        actions[env_id].append(output[env_id]['action'])
                actions = to_ndarray(actions)
                timesteps = self.env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                self.env_info[env_id]['step'] += 1
                self.total_envstep_count += 1
                ctx.envstep = self.total_envstep_count
                with self._timer:
                    for policy_id, policy in enumerate(ctx._policy):
                        policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                        policy_timestep = type(timestep)(*policy_timestep_data)
                        transition = ctx._policy[policy_id].process_transition(
                            self.obs_pool[env_id][policy_id], self.policy_output_pool[env_id][policy_id],
                            policy_timestep
                        )
                        transition['collect_iter'] = ctx.train_iter
                        self.traj_buffer[env_id][policy_id].append(transition)
                        # prepare data
                        if timestep.done:
                            transitions = to_tensor_transitions(self.traj_buffer[env_id][policy_id])
                            if self.cfg.get_train_sample:
                                train_sample = ctx._policy[policy_id].get_train_sample(transitions)
                                return_data[policy_id].extend(train_sample)
                            else:
                                return_data[policy_id].append(transitions)
                            self.traj_buffer[env_id][policy_id].clear()

                self.env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self.total_episode_count += 1
                    info = {
                        'reward0': timestep.info[0]['final_eval_reward'],
                        'reward1': timestep.info[1]['final_eval_reward'],
                        'time': self.env_info[env_id]['time'],
                        'step': self.env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    self.episode_info.append(info)
                    for i, p in enumerate(ctx._policy):
                        p.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
                    for policy_id in range(2):
                        return_info[policy_id].append(timestep.info[policy_id])
            if collected_episode >= ctx.n_episode:
                break
        # log
        ### TODO: how to deal with log here?
        # self._output_log(ctx.train_iter)
        ctx.train_data = return_data
        ctx.episode_info = return_info


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
