from typing import TYPE_CHECKING
from easydict import EasyDict
import treetensor.torch as ttorch

from ding.policy import get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext

import numpy as np
import torch


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


class EnvpoolStepCollector:

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.COLLECTOR):
            return task.void()
        return super(EnvpoolStepCollector, cls).__new__(cls)

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

        self._ready_obs_receive = {}
        self._ready_obs_send = {}
        self._ready_action_send = {}
        self._trajectory = {i: [] for i in range(env.env_num)}
        self._nsteps = self.cfg.policy.nstep if hasattr(self.cfg.policy, 'nstep') else 1
        self._discount_ratio_list = [self.cfg.policy.discount_factor ** (i + 1) for i in range(self._nsteps)]
        self._nsteps_range = list(range(1, self._nsteps))
        self.policy = policy
        self.random_collect_size = random_collect_size

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
            random = True
        else:
            target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len
            random = False

        if self.env.closed:
            self._ready_obs_receive = self.env.launch()

        counter = 0

        while True:
            if len(self._ready_obs_receive.keys()) > 0:
                if random:
                    action_to_send = {
                        i: {
                            "action": np.array([self.env.action_space.sample()])
                        }
                        for i in self._ready_obs_receive.keys()
                    }
                else:
                    action_by_policy = self.policy.forward(self._ready_obs_receive, **ctx.collect_kwargs)

                    if isinstance(list(action_by_policy.values())[0]['action'], torch.Tensor):
                        # transfer to numpy
                        action_to_send = {
                            i: {
                                "action": action_by_policy[i]['action'].cpu().numpy()
                            }
                            for i in action_by_policy.keys()
                        }
                    else:
                        action_to_send = action_by_policy
                self._ready_obs_send.update(self._ready_obs_receive)
                self._ready_obs_receive = {}
                self._ready_action_send.update(action_to_send)

                action_send = np.array([action_to_send[i]['action'] for i in action_to_send.keys()])
                if action_send.ndim == 2 and action_send.shape[1] == 1:
                    action_send = action_send.squeeze(1)
                env_id_send = np.array(list(action_to_send.keys()))
                self.env.send_action(action_send, env_id_send)

            next_obs, rew, done, info = self.env.receive_data()
            env_id_receive = info['env_id']
            counter += len(env_id_receive)
            self._ready_obs_receive.update({i: next_obs[i] for i in range(len(next_obs))})

            #todo
            for i in range(len(env_id_receive)):
                current_reward = rew[i]
                if self._nsteps > 1:
                    self._trajectory[env_id_receive[i]].append(
                        {
                            'obs': self._ready_obs_send[env_id_receive[i]],
                            'action': self._ready_action_send[env_id_receive[i]]['action'],
                            'next_obs': next_obs[i],
                            # n-step reward
                            'reward': [current_reward],
                            'done': done[i],
                        }
                    )
                else:
                    self._trajectory[env_id_receive[i]].append(
                        {
                            'obs': self._ready_obs_send[env_id_receive[i]],
                            'action': self._ready_action_send[env_id_receive[i]]['action'],
                            'next_obs': next_obs[i],
                            # n-step reward
                            'reward': current_reward,
                            'done': done[i],
                        }
                    )

                if self._nsteps > 1:
                    if done[i] is False and counter < target_size:
                        reverse_record_position = min(self._nsteps, len(self._trajectory[env_id_receive[i]]))
                        real_reverse_record_position = reverse_record_position

                        for j in range(1, reverse_record_position + 1):
                            if j == 1:
                                pass
                            else:
                                if self._trajectory[env_id_receive[i]][-j]['done'] is True:
                                    real_reverse_record_position = j - 1
                                    break
                                else:
                                    self._trajectory[env_id_receive[i]][-j]['reward'].append(current_reward)

                        if real_reverse_record_position == self._nsteps:
                            self._trajectory[env_id_receive[i]
                                             ][-real_reverse_record_position]['next_n_obs'] = next_obs[i]
                            self._trajectory[env_id_receive[i]][-real_reverse_record_position][
                                'value_gamma'] = self._discount_ratio_list[real_reverse_record_position - 1]

                    else:  # done[i] is True or counter >= target_size

                        reverse_record_position = min(self._nsteps, len(self._trajectory[env_id_receive[i]]))
                        real_reverse_record_position = reverse_record_position

                        for j in range(1, reverse_record_position + 1):
                            if j == 1:
                                self._trajectory[env_id_receive[i]][-j]['reward'].extend(
                                    [
                                        0.0 for _ in
                                        range(self._nsteps - len(self._trajectory[env_id_receive[i]][-j]['reward']))
                                    ]
                                )
                                self._trajectory[env_id_receive[i]][-j]['next_n_obs'] = next_obs[i]
                                self._trajectory[env_id_receive[i]][-j]['value_gamma'] = self._discount_ratio_list[j -
                                                                                                                   1]
                            else:
                                if self._trajectory[env_id_receive[i]][-j]['done'] is True:
                                    real_reverse_record_position = j
                                    break
                                else:
                                    self._trajectory[env_id_receive[i]][-j]['reward'].append(current_reward)
                                    self._trajectory[env_id_receive[i]][-j]['reward'].extend(
                                        [
                                            0.0 for _ in range(
                                                self._nsteps - len(self._trajectory[env_id_receive[i]][-j]['reward'])
                                            )
                                        ]
                                    )
                                    self._trajectory[env_id_receive[i]][-j]['next_n_obs'] = next_obs[i]
                                    self._trajectory[env_id_receive[i]][-j]['value_gamma'] = self._discount_ratio_list[
                                        j - 1]

                else:
                    self._trajectory[env_id_receive[i]][-1]['value_gamma'] = self._discount_ratio_list[0]

            if counter >= target_size:
                break

        ctx.trajectories = []
        for i in range(self.env.env_num):
            ctx.trajectories.extend(self._trajectory[i])
            self._trajectory[i] = []
        ctx.env_step += len(ctx.trajectories)


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
