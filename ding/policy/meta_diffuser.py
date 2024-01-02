from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple, defaultdict
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.policy import Policy
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, DatasetNormalizer
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn

@POLICY_REGISTRY.register('metadiffuser')
class MDPolicy(Policy):
    r"""
    Overview:
        Implicit Meta Diffuser
        https://arxiv.org/pdf/2305.19923.pdf

    """
    config = dict(
        type='pd',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        nstep=1,
        # normalizer type
        normalizer='GaussianNormalizer',
        model=dict(
            dim=32,
            obs_dim=17,
            action_dim=6,
            diffuser_cfg=dict(
                # the type of model
                # config of model
                model_cfg=dict(
                    # model dim, In GaussianInvDynDiffusion, it is obs_dim. In others, it is obs_dim + action_dim
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 2, 4, 8],
                    # whether use return as a condition
                    returns_condition=True,
                    condition_dropout=0.1,
                    # whether use calc energy
                    calc_energy=False,
                    kernel_size=5,
                    # whether use attention
                    attention=False,
                ),
                # horizon of tarjectory which generated by model
                horizon=80,
                # timesteps of diffusion
                n_timesteps=1000,
                # hidden dim of action model
                # Whether predict epsilon
                predict_epsilon=True,
                # discount of loss
                loss_discount=1.0,
                # whether clip denoise
                clip_denoised=False,
                action_weight=10,
            ),
            reward_cfg=dict(
                # the type of model
                model='TemporalValue',
                # config of model
                model_cfg=dict(
                    horizon=4,
                    # model dim, In GaussianInvDynDiffusion, it is obs_dim. In others, it is obs_dim + action_dim
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 2, 4, 8],
                    # whether use calc energy
                    kernel_size=5,
                ),
                # horizon of tarjectory which generated by model
                horizon=80,
                # timesteps of diffusion
                n_timesteps=1000,
                # hidden dim of action model
                predict_epsilon=True,
                # discount of loss
                loss_discount=1.0,
                # whether clip denoise
                clip_denoised=False,
                action_weight=1.0,
            ),
            horizon=80,
            # guide_steps for p sample
            n_guide_steps=2,
            # scale of grad for p sample
            scale=1,
            # t of stopgrad for p sample
            t_stopgrad=2,
            # whether use std as a scale for grad
            scale_grad_by_std=True,
        ),
        learn=dict(

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=100,

            # (float type) learning_rate_q: Learning rate for model.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate=3e-4,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,

            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            gradient_accumulate_every=2,
            # train_epoch = train_epoch * gradient_accumulate_every
            train_epoch=60000,
            # batch_size of every env when eval
            plan_batch_size=64,

            # step start update target model and frequence
            step_start_update_target=2000,
            update_target_freq=10,
            # update weight of target net
            target_weight=0.995,
            value_step=200e3,

            # dataset weight include returns
            include_returns=True,

            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'md', ['ding.model.template.diffusion']
    
    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.action_dim = self._cfg.model.diffuser_model_cfg.action_dim
        self.obs_dim = self._cfg.model.diffuser_model_cfg.obs_dim
        self.n_timesteps = self._cfg.model.diffuser_model_cfg.n_timesteps
        self.gradient_accumulate_every = self._cfg.learn.gradient_accumulate_every
        self.plan_batch_size = self._cfg.learn.plan_batch_size
        self.gradient_steps = 1
        self.update_target_freq = self._cfg.learn.update_target_freq
        self.step_start_update_target = self._cfg.learn.step_start_update_target
        self.target_weight = self._cfg.learn.target_weight
        self.value_step = self._cfg.learn.value_step
        self.horizon = self._cfg.model.diffuser_model_cfg.horizon
        self.include_returns = self._cfg.learn.include_returns
        self.eval_batch_size = self._cfg.learn.eval_batch_size
        self.warm_batch_size = self._cfg.learn.warm_batch_size

        self._plan_optimizer = Adam(
            self._model.diffuser.model.parameters(),
            lr=self._cfg.learn.learning_rate,
        )

        self._pre_train_optimizer = Adam(
            list(self._model.reward_model.model.parameters()) + list(self._model.embed.parameters()) \
                  + list(self._model.dynamic_model.parameters()),
            lr=self._cfg.learn.learning_rate,
        )

        self._gamma = self._cfg.learn.discount_factor

        self._target_model = copy.deepcopy(self._model)

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    def _forward_learn(self, data: List[torch.Tensor]) -> Dict[str, Any]:
        loss_dict = {}

        if self._cuda:
            data = to_device(data, self._device)
        timesteps, obs, acts, rewards, rtg, masks, cond_id, cond_vals = data
        obs, next_obs = obs[:-1], obs[1:]
        acts = acts[:-1]
        rewards = rewards[:-1]
        conds = {cond_id: cond_vals}
        

        self._learn_model.train()
        pre_traj = torch.cat([acts, obs, rewards, next_obs], dim=1)
        target = torch.cat([next_obs, rewards], dim=1)
        traj = torch.cat([acts, obs], dim=1)

        batch_size = len(traj)
        t = torch.randint(0, self.n_timesteps, (batch_size, ), device=traj.device).long()
        state_loss, reward_loss = self._learn_model.pre_train_loss(pre_traj, target, t, conds)
        loss_dict = {'state_loss': state_loss, 'reward_loss': reward_loss}
        total_loss = state_loss + reward_loss

        self._pre_train_optimizer.zero()
        total_loss.backward()
        self._pre_train_optimizer.step()
        self.update_model_average(self._target_model, self._learn_model)

        diffuser_loss = self._learn_model.diffuser_loss(traj, conds, t)
        self._plan_optimizer.zero()
        diffuser_loss.backward()
        self._plan_optimizer.step()
        self.update_model_average(self._target_model, self._learn_model)

        return loss_dict



    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            if old_weight is None:
                ma_params.data = up_weight
            else:
                old_weight * self.target_weight + (1 - self.target_weight) * up_weight

    def init_dataprocess_func(self, dataloader: torch.utils.data.Dataset):
        self.dataloader = dataloader

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'diffuse_loss',
            'reward_loss',
            'dynamic_loss',
            'max_return',
            'min_return',
            'mean_return',
            'a0_loss',
        ]
    
    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
                'model': self._learn_model.state_dict(),
                'target_model': self._target_model.state_dict(),
                'plan_optimizer': self._plan_optimizer.state_dict(),
                'pre_train_optimizer': self._pre_train_optimizer.state_dict(),
            }
    
    def _init_eval(self):
        self._eval_model = model_wrap(self._target_model, wrapper_name='base')
        self._eval_model.reset()
        self.task_id = [0] * self.eval_batch_size
        

        obs, acts, rewards, cond_ids, cond_vals = \
            self.dataloader.get_pretrain_data(self.task_id[0], self.warm_batch_size * self.eval_batch_size)
        obs = to_device(obs, self._device)
        acts = to_device(acts, self._device)
        rewards = to_device(rewards, self._device)
        cond_vals = to_device(cond_vals, self._device)
        
        obs, next_obs = obs[:-1], obs[1:]
        acts = acts[:-1]
        rewards = rewards[:-1]
        pre_traj = torch.cat([acts, obs, next_obs, rewards], dim=1)
        target = torch.cat([next_obs, rewards], dim=1)
        batch_size = len(pre_traj)
        conds = {cond_ids: cond_vals}

        t = torch.randint(0, self.n_timesteps, (batch_size, ), device=pre_traj.device).long()
        state_loss, reward_loss = self._learn_model.pre_train_loss(pre_traj, target, t, conds)
        total_loss = state_loss + reward_loss
        self._pre_train_optimizer.zero()
        total_loss.backward()
        self._pre_train_optimizer.step()
        self.update_model_average(self._target_model, self._learn_model)

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))

        self._eval_model.eval()
        obs = []
        for i in range(self.eval_batch_size):
            obs.append(self.dataloader.normalize(data, 'observations', self.task_id[i]))

        with torch.no_grad():
            obs = torch.tensor(obs)
            if self._cuda:
                obs = to_device(obs, self._device)
            conditions = {0: obs}
            action = self._eval_model.get_eval(conditions, self.plan_batch_size)
            if self._cuda:
                action = to_device(action, 'cpu')
            for i in range(self.eval_batch_size):
                action[i] = self.dataloader.unnormalize(action, 'actions', self.task_id[i])
            action = torch.tensor(action).to('cpu') 
        output = {'action': action}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self.task_id[data_id] += 1

        obs, acts, rewards, cond_ids, cond_vals = \
            self.dataloader.get_pretrain_data(self.task_id[data_id], self.warm_batch_size)
        obs = to_device(obs, self._device)
        acts = to_device(acts, self._device)
        rewards = to_device(rewards, self._device)
        cond_vals = to_device(cond_vals, self._device)

        obs, next_obs = obs[:-1], obs[1:]
        acts = acts[:-1]
        rewards = rewards[:-1]
        pre_traj = torch.cat([acts, obs, next_obs, rewards], dim=1)
        target = torch.cat([next_obs, rewards], dim=1)
        batch_size = len(pre_traj)
        conds = {cond_ids: cond_vals}

        t = torch.randint(0, self.n_timesteps, (batch_size, ), device=pre_traj.device).long()
        state_loss, reward_loss = self._learn_model.pre_train_loss(pre_traj, target, t, conds)
        total_loss = state_loss + reward_loss
        self._pre_train_optimizer.zero()
        total_loss.backward()
        self._pre_train_optimizer.step()
        self.update_model_average(self._target_model, self._learn_model)