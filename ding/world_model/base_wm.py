from abc import ABC, abstractmethod

import torch
import copy
from easydict import EasyDict

from ding.utils import deep_merge_dicts
from .utils import get_rollout_length_scheduler

class WorldModel(ABC):

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

    def __init__(self, cfg, env, tb_logger):
        self.cfg = cfg
        # will be potentially used in derived class
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
        # can not call default_config recursively 
        # because config will be overwritten by subclasses
        merge_cfg = EasyDict(cfg_type=cls.__name__ + 'Dict')
        while cls != ABC:
            merge_cfg = deep_merge_dicts(merge_cfg, cls.config)
            cls = cls.__base__
        return merge_cfg

    def should_train(self, envstep):
        if (envstep - self.last_train_step) < self.train_freq:
            return False
        return True

    def should_eval(self, envstep):
        if (envstep - self.last_eval_step) < self.eval_freq or self.last_train_step == 0:
            return False
        return True

    @abstractmethod
    def train(self, env_buffer, envstep, train_iter):
        raise NotImplementedError

    @abstractmethod
    def eval(self, env_buffer, envstep, train_iter):
        raise NotImplementedError

    @abstractmethod
    def step(self, obs, action):
        # reward, next_obs, done
        raise NotImplementedError



class DynaWorldModel(WorldModel, ABC):
    """Dyna style - reuse model rollout"""

    # rollout_scheduler
    config = dict(
        other=dict(
            real_ratio=0.05,
            rollout_retain=4,
            rollout_batch_size=100000,
        )
    )

    def __init__(self, cfg, env, tb_logger):
        super().__init__(cfg, env, tb_logger)
        self.real_ratio = cfg.other.real_ratio
        self.rollout_batch_size = cfg.other.rollout_batch_size
        self.rollout_retain = cfg.other.rollout_retain
        self.buffer_size_scheduler = \
            lambda x: self.rollout_length_scheduler(x) \
                * self.rollout_batch_size * self.rollout_retain

    def sample(self, env_buffer, img_buffer, batch_size, train_iter):
        env_batch_size = int(batch_size * self.real_ratio)
        img_batch_size = batch_size - env_batch_size
        env_data = env_buffer.sample(env_batch_size, train_iter)
        img_data = img_buffer.sample(img_batch_size, train_iter)
        train_data = env_data + img_data
        return train_data

    def fill_img_buffer(self, policy, env_buffer, img_buffer, envstep, cur_learner_iter):
        """
        Overview:
            This function samples from the replay_buffer, rollouts to generate new data,
            and push them into the imagine_buffer.
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
    """Dreamer style - on the fly rollout"""

    # TODO: use scheduler to set horizon
    def rollout(self, obs, actor_fn, envstep):
        # actor_fn is batch_mode policy.forward with no collate and decollate
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
    def __init__(self, cfg, env, tb_logger):
        DynaWorldModel.__init__(self, cfg, env, tb_logger)
        DreamWorldModel.__init__(self, cfg, env, tb_logger)
