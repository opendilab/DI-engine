from distutils.log import info
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
from easydict import EasyDict
from ding.policy import Policy, get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList, battle_inferencer, battle_rolloutor

# if TYPE_CHECKING:
from ding.framework import OnlineRLContext

from ding.utils import EasyTimer
from ding.worker.collector.base_serial_collector import CachePool


class BattleCollector:
    def __init__(
            self, 
            cfg: EasyDict, 
            env: BaseEnvManager, 
            n_rollout_samples: int
    ):
        self.cfg = cfg
        self._timer = EasyTimer()
        self.end_flag = False
        # self._reset(env)
        self.env = env
        self.env_num = self.env.env_num

        self.obs_pool = CachePool('obs', self.env_num, deepcopy=self.cfg.deepcopy_obs)
        self.policy_output_pool = CachePool('policy_output', self.env_num)
        self.env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self.env_num)}

        self.episode_info = []
        self.total_envstep_count = 0
        self.total_episode_count = 0
        self.end_flag = False
        self.n_rollout_samples = n_rollout_samples

        self._battle_inferencer = task.wrap(battle_inferencer(self.cfg, self.env, self.obs_pool, self.policy_output_pool))
        self._battle_rolloutor = task.wrap(battle_rolloutor(self.cfg, self.obs_pool, self.policy_output_pool))
    
    def _reset_stat(self, env_id: int, ctx: OnlineRLContext) -> None:
        """
        Overview:
            Reset the collector's state. Including reset the traj_buffer, obs_pool, policy_output_pool\
                and env_info. Reset these states according to env_id. You can refer to base_serial_collector\
                to get more messages.
        Arguments:
            - env_id (:obj:`int`): the id where we need to reset the collector's state
        """
        for i in range(ctx.agent_num):
            ctx.traj_buffer[env_id][i].clear()
        self.obs_pool.reset(env_id)
        self.policy_output_pool.reset(env_id)
        self.env_info[env_id] = {'time': 0., 'step': 0}

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  ctx.train_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
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
        
        if self.env.closed:
            self.env.launch()

        ctx.collected_episode = 0
        ctx.train_data = [[] for _ in range(ctx.agent_num)]
        ctx.episode_info = [[] for _ in range(ctx.agent_num)]
        ctx.ready_env_id = set()
        ctx.remain_episode = ctx.n_episode
        while True:
            with self._timer:
                self._battle_inferencer(ctx)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(ctx.timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in ctx.timesteps.items():
                self.env_info[env_id]['step'] += 1
                self.total_envstep_count += 1
                ctx.envstep = self.total_envstep_count
                ctx.env_id = env_id
                ctx.timestep = timestep
                with self._timer:
                    self._battle_rolloutor(ctx)

                self.env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self.total_episode_count += 1
                    info = {
                        'time': self.env_info[env_id]['time'],
                        'step': self.env_info[env_id]['step'],
                    }
                    for policy_id in range(ctx.agent_num):
                        info['reward'+str(policy_id)] = timestep.info[policy_id]['final_eval_reward']
                    ctx.collected_episode += 1
                    self.episode_info.append(info)
                    for i, p in enumerate(ctx.policies):
                        p.reset([env_id])
                    self._reset_stat(env_id, ctx)
                    ctx.ready_env_id.remove(env_id)
                    for policy_id in range(ctx.agent_num):
                        ctx.episode_info[policy_id].append(timestep.info[policy_id])
            if ctx.collected_episode >= ctx.n_episode:
                break
        

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
