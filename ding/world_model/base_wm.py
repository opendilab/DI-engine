
from typing import Tuple, Callable
from collections import namedtuple
from abc import ABC, abstractmethod

import torch
from torch import Tensor, tensor
from easydict import EasyDict

from ding.worker import IBuffer
from ding.envs import BaseEnv
from ding.utils import deep_merge_dicts
from .utils import get_rollout_length_scheduler

class WorldModel(ABC):
    """
    Overview:
        Abstract baseclass for world model.

    Interfaces:
        should_train, should_eval, train, eval, step
    """

    config = dict(
        train_freq=250,  # w.r.t environment step
        eval_freq=20,    # w.r.t environment step
        cuda=True,
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=20000,
            rollout_end_step=150000,
            rollout_length_min=1,
            rollout_length_max=25,
        )
    )

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        self.cfg = cfg
        self.env = env
        self.tb_logger = tb_logger

        self._cuda = cfg.cuda
        self.train_freq = cfg.train_freq
        self.eval_freq = cfg.eval_freq
        self.rollout_length_scheduler = \
            get_rollout_length_scheduler(cfg.rollout_length_scheduler)

        self.last_train_step = 0
        self.last_eval_step = 0
    
    @classmethod
    def default_config(cls: type) -> EasyDict:
        # can not call default_config() recursively 
        # because config will be overwritten by subclasses
        merge_cfg = EasyDict(cfg_type=cls.__name__ + 'Dict')
        while cls != ABC:
            merge_cfg = deep_merge_dicts(merge_cfg, cls.config)
            cls = cls.__base__
        return merge_cfg

    def should_train(self, envstep: int):
        """
        Overview:
            Check whether need to train world model.
        """
        if (envstep - self.last_train_step) < self.train_freq:
            return False
        return True

    def should_eval(self, envstep: int):
        """
        Overview:
            Check whether need to evaluate world model.
        """
        if (envstep - self.last_eval_step) < self.eval_freq or self.last_train_step == 0:
            return False
        return True

    @abstractmethod
    def train(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        """
        Overview:
            Train world model using data from env_buffer.
        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer which collects real environment steps.
            - envstep (:obj:`int`): the current number of environment steps in real environment.
            - train_iter (:obj:`int`): the current number of policy training iterations.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        """
        Overview:
            Evaluate world model using data from env_buffer.
        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps.
            - envstep (:obj:`int`): the current number of environment steps in real environment.
            - train_iter (:obj:`int`): the current number of policy training iterations.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Overview:
            Take one step in world model.
        Arguments:
            - obs (:obj:`torch.Tensor`): current observations S_t.
            - action (:obj:`torch.Tensor`): current actions A_t. 
        Returns:
            - reward (:obj:`torch.Tensor`): rewards R_t.
            - next_obs (:obj:`torch.Tensor`): next observations S_t+1.
            - done (:obj:`torch.Tensor`): whether the episodes ends.
        Shapes:
            B: batch size
            O: observation dimension
            A: action dimension

            - obs:      [B, O]
            - action:   [B, A]
            - reward:   [B, ]
            - next_obs: [B, O]
            - done:     [B, ]
        """
        raise NotImplementedError



class DynaWorldModel(WorldModel, ABC):
    """
    Overview:
        Dyna-style world model which stores and reuses imagination rollout in the imagination buffer.

    Interfaces:
        sample, fill_img_buffer, should_train, should_eval, train, eval, step
    """

    # rollout_scheduler
    config = dict(
        other=dict(
            real_ratio=0.05,
            rollout_retain=4,
            rollout_batch_size=100000,
        )
    )

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        super().__init__(cfg, env, tb_logger)
        self.real_ratio = cfg.other.real_ratio
        self.rollout_batch_size = cfg.other.rollout_batch_size
        self.rollout_retain = cfg.other.rollout_retain
        self.buffer_size_scheduler = \
            lambda x: self.rollout_length_scheduler(x) \
                * self.rollout_batch_size * self.rollout_retain

    def sample(
            self, 
            env_buffer: IBuffer, 
            img_buffer: IBuffer, 
            batch_size: int, 
            train_iter: int
    ) -> dict:
        """
        Overview:
            Sample from the combination of environment buffer and imagination buffer with
            certain ratio to generate batched data for policy training.
        Arguments:
            - policy (:obj:`namedtuple`): policy in collect mode.
            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps.
            - img_buffer (:obj:`IBuffer`): the buffer that collects imagination steps.
            - batch_size (:obj:`int`): the batch size for policy training. 
            - learner_iter (:obj:`int`): the current number of policy training iterations.
        Returns:
            - data (:obj:`int`): the training data for policy training.
        """
        env_batch_size = int(batch_size * self.real_ratio)
        img_batch_size = batch_size - env_batch_size
        env_data = env_buffer.sample(env_batch_size, train_iter)
        img_data = img_buffer.sample(img_batch_size, train_iter)
        train_data = env_data + img_data
        return train_data

    def fill_img_buffer(
            self, 
            policy: namedtuple, 
            env_buffer: IBuffer, 
            img_buffer: IBuffer, 
            envstep: int, 
            cur_learner_iter: int
    ):
        """
        Overview:
            Sample from the env_buffer, rollouts to generate new data, and push them into the img_buffer.
        Arguments:
            - policy (:obj:`namedtuple`): policy in collect mode.
            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps.
            - img_buffer (:obj:`IBuffer`): the buffer that collects imagination steps.
            - envstep (:obj:`int`): the current number of environment steps in real environment.
            - cur_learner_iter (:obj:`int`): the current number of policy training iterations.
        """
        from ding.torch_utils import to_tensor
        from ding.envs import BaseEnvTimestep
        from ding.worker.collector.base_serial_collector import to_tensor_transitions

        def step(obs, act):
            # This function has the same input and output format as env manager's step
            data_id = list(obs.keys())
            obs = torch.stack([obs[id] for id in data_id], dim=0)
            act = torch.stack([act[id] for id in data_id], dim=0)
            with torch.no_grad():
                rewards, next_obs, terminals = self.step(obs, act)
            # terminals = self.termination_fn(next_obs)
            timesteps = {
                id: BaseEnvTimestep(n, r, d, {})
                for id, n, r, d in zip(data_id, next_obs.cpu().numpy(), rewards.cpu().numpy(), terminals.cpu().numpy())
            }
            return timesteps

        # set rollout length
        rollout_length = self.rollout_length_scheduler(envstep)
        # load data
        data = env_buffer.sample(self.rollout_batch_size, cur_learner_iter, replace=True)
        obs = {id: data[id]['obs'] for id in range(len(data))}
        # rollout
        buffer = [[] for id in range(len(obs))]
        new_data = []
        for i in range(rollout_length):
            # get action
            obs = to_tensor(obs, dtype=torch.float32)
            policy_output = policy.forward(obs)
            actions = {id: output['action'] for id, output in policy_output.items()}
            # predict next obs and reward
            # timesteps = self.step(obs, actions, env_model)
            timesteps = step(obs, actions)
            obs_new = {}
            for id, timestep in timesteps.items():
                transition = policy.process_transition(obs[id], policy_output[id], timestep)
                transition['collect_iter'] = cur_learner_iter
                buffer[id].append(transition)
                if not timestep.done:
                    obs_new[id] = timestep.obs
                if timestep.done or i + 1 == rollout_length:
                    transitions = to_tensor_transitions(buffer[id])
                    train_sample = policy.get_train_sample(transitions)
                    new_data.extend(train_sample)
            if len(obs_new) == 0:
                break
            obs = obs_new

        img_buffer.push(new_data, cur_collector_envstep=envstep)


class DreamWorldModel(WorldModel, ABC):
    """
    Overview:
        Dreamer-style world model which uses each imagination rollout only once
        and backpropagate through time(rollout) to optimize policy. 

    Interfaces:
        rollout, should_train, should_eval, train, eval, step
    """

    def rollout(
            self, 
            obs: Tensor, 
            actor_fn: Callable[[Tensor], Tuple[Tensor, Tensor]], 
            envstep: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Overview:
            Generate batched imagination rollouts starting from the current observations.
            This function is useful for value gradients where the policy is optimized by BPTT.
        Arguments:
            - obs (:obj:`Tensor`): the current observations S_t.
            - actor_fn (:obj:`Callable`): the unified API (A_t, H_t) = pi(S_t).
            - envstep (:obj:`int`): the current number of environment steps in real environment.
        Returns:
            - obss (:obj:`Tensor`):        S_t,  ...  S_t+n
            - actions (:obj:`Tensor`):     A_t,  ..., A_t+n
            - rewards (:obj:`Tensor`):     R_t,  ..., R_t+n-1
            - aug_rewards (:obj:`Tensor`): H_t', ..., H_t+n, this can be entropy bonus as in SAC,
                                                otherwise it should be a zero tensor.
            - dones (:obj:`Tensor`):       done_t, ..., done_t+n
        Shapes:
            N: time step
            B: batch size
            O: observation dimension
            A: action dimension

            - obss:        [N+1, B, O], where obss[0] are the real observations.
            - actions:     [N+1, B, A]
            - rewards:     [N,   B]
            - aug_rewards: [N+1, B] 
            - dones:       [N+1, B]

        .. note::
            - The rollout length is determined by rollout length scheduler

            - actor_fn's inputs and outputs shape are similar to WorldModel.step()
        """
        horizon = self.rollout_length_scheduler(envstep)
        obss        = [obs]
        actions     = []
        rewards     = []
        aug_rewards = []    # -temperature*logprob
        dones       = [torch.zeros_like(obs.sum(-1))]
        for _ in range(horizon):
            action, aug_reward = actor_fn(obs)
            # done: probability of termination
            reward, obs, done = self.step(obs, action)
            if len(reward.shape) == 2:
                reward = reward.squeeze(1)
            if len(done.shape) == 2:
                done = done.squeeze(1)
            reward = reward + aug_reward
            obss.append(obs)
            actions.append(action)
            rewards.append(reward)
            aug_rewards.append(aug_reward)
            dones.append(done)
        action, aug_reward = actor_fn(obs)
        actions.append(action)
        aug_rewards.append(aug_reward)
        return (
            torch.stack(obss), 
            torch.stack(actions), 
            # rewards is an empty list when horizon=0
            torch.stack(rewards),
            torch.stack(aug_rewards), 
            torch.stack(dones)
        )


class HybridWorldModel(DynaWorldModel, DreamWorldModel):
    """
    Overview:
        The hybrid model that combines reused and on-the-fly rollouts.

    Interfaces:
        rollout, sample, fill_img_buffer, should_train, should_eval, train, eval, step
    """
    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        DynaWorldModel.__init__(self, cfg, env, tb_logger)
        DreamWorldModel.__init__(self, cfg, env, tb_logger)
