from typing import TYPE_CHECKING
from easydict import EasyDict
import treetensor.torch as ttorch

from ding.policy import get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


class StepCollector:
    """
    Overview:
        The class of the collector running by steps, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.COLLECTOR):
            return task.void()
        return super(StepCollector, cls).__new__(cls)

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
        self._inferencer = task.wrap(inferencer(cfg.seed, policy, env))
        self._rolloutor = task.wrap(rolloutor(policy, env, self._transitions))

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
            current_inferencer = task.wrap(inferencer(self.cfg.seed, random_policy, self.env))
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


class PPOFStepCollector:
    """
    Overview:
        The class of the collector running by steps, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.COLLECTOR):
            return task.void()
        return super(PPOFStepCollector, cls).__new__(cls)

    def __init__(self, seed: int, policy, env: BaseEnvManager, n_sample: int, unroll_len: int = 1) -> None:
        """
        Arguments:
            - seed (:obj:`int`): Random seed.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
        """
        self.env = env
        self.env.seed(seed)
        self.policy = policy
        self.n_sample = n_sample
        self.unroll_len = unroll_len
        self._transitions = TransitionList(self.env.env_num)
        self._env_episode_id = [_ for _ in range(env.env_num)]
        self._current_id = env.env_num

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing \
                the target number of steps.
        Input of ctx:
            - env_step (:obj:`int`): The env steps which will increase during collection.
        """
        device = self.policy._device
        old = ctx.env_step
        target_size = self.n_sample * self.unroll_len

        if self.env.closed:
            self.env.launch()

        while True:
            obs = ttorch.as_tensor(self.env.ready_obs).to(dtype=ttorch.float32)
            obs = obs.to(device)
            inference_output = self.policy.collect(obs, **ctx.collect_kwargs)
            inference_output = inference_output.cpu()
            action = inference_output.action.numpy()
            timesteps = self.env.step(action)
            ctx.env_step += len(timesteps)

            obs = obs.cpu()
            for i, timestep in enumerate(timesteps):
                transition = self.policy.process_transition(obs[i], inference_output[i], timestep)
                transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
                transition.env_data_id = ttorch.as_tensor([self._env_episode_id[timestep.env_id]])
                self._transitions.append(timestep.env_id, transition)
                if timestep.done:
                    self.policy.reset([timestep.env_id])
                    self._env_episode_id[timestep.env_id] = self._current_id
                    self._current_id += 1
                    ctx.env_episode += 1

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
        self._inferencer = task.wrap(inferencer(cfg.seed, policy, env))
        self._rolloutor = task.wrap(rolloutor(policy, env, self._transitions))

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
