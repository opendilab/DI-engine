"""The code is adapted from https://github.com/nikhilbarhate99/min-decision-transformer
"""

from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
from torch.distributions import Normal, Independent
from ding.torch_utils import Adam, to_device
from ditk import logging
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils.data.dataset import D4RLTrajectoryDataset
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from datetime import datetime
from ding.torch_utils import one_hot
import numpy as np
import torch.nn.functional as F
import torch
import gym
import copy
import os
import csv
from .base_policy import Policy


@POLICY_REGISTRY.register('dt')
class DTPolicy(Policy):
    r"""
    Overview:
        Policy class of Decision Transformer algorithm in discrete environments.
        Paper link: https://arxiv.org/abs/2106.01345
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dt',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        obs_shape=4,
        action_shape=2,
        # encoder_hidden_size_list=[128, 128, 64],
        dataset='medium',  # medium / medium-replay / medium-expert
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        num_eval_ep=10,  # num of evaluation episodes
        batch_size=64,  # training batch size
        wt_decay=1e-4,
        warmup_steps=10000,
        max_train_iters=200,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        dropout_p=0.1,
        log_dir='/mnt/nfs/luyd/DI-engine/dizoo/box2d/lunarlander/dt_log_1000eps',   
        learn=dict(

            dataset_path='/mnt/nfs/luyd/DI-engine/dizoo/box2d/lunarlander/offline_data/dt_data/dqn_data_1000eps.pkl',  # TODO
            # batch_size=64,
            learning_rate=1e-4,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
        ),
        # collect_mode config
        collect=dict(),
        eval=dict(),
        # other config
        other=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dt', ['ding.model.template.decision_transformer']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        self.env_name = self._cfg.env_name
        # rtg_scale: scale of `return to go`
        # rtg_target: max target of `return to go`
        # Our goal is normalize `return to go` to (0, 1), which will favour the covergence.
        # As a result, we usually set rtg_scale == rtg_target.
        self.rtg_scale = self._cfg.rtg_scale  # normalize returns to go 
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.max_eval_ep_len = self._cfg.max_eval_ep_len  # max len of one episode
        self.num_eval_ep = self._cfg.num_eval_ep  # num of evaluation episodes

        lr = self._cfg.learn.learning_rate  # learning rate
        wt_decay = self._cfg.wt_decay  # weight decay
        warmup_steps = self._cfg.warmup_steps  # warmup steps for lr scheduler

        self.context_len = self._cfg.context_len  # K in decision transformer

        # # load data from this file
        # dataset_path = f'{self._cfg.dataset_dir}/{env_d4rl_name}.pkl'

        # training and evaluation device
        self.device = torch.device(self._device)

        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim

        self._learn_model = self._model
        self._optimizer = torch.optim.AdamW(self._learn_model.parameters(), lr=lr, weight_decay=wt_decay)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        self.max_env_score = -1.0

    def _forward_learn(self, data: list) -> Dict[str, Any]:
        r"""
            Overview:
                Forward and backward function of learn mode.
            Arguments:
                - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
            Returns:
                - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """

        self._learn_model.train()

        data = [[i[j] for i in data] for j in range(len(data[0]))]
        timesteps, states, actions, returns_to_go, traj_mask = data
        timesteps = torch.stack(timesteps).to(self.device)  # B x T
        states = torch.stack(states).to(self.device)  # B x T x state_dim
        actions = torch.stack(actions).to(self.device)  # B x T x act_dim
        returns_to_go = torch.stack(returns_to_go).to(self.device)  # B x T x 1
        traj_mask = torch.stack(traj_mask).to(self.device)  # B x T
        action_target = torch.clone(actions).detach().to(self.device)

        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        # if discrete
        if not self._cfg.model.continuous and self.cfg.env_type != 'atari':
            actions = one_hot(actions.squeeze(-1), num=self.act_dim)

        state_preds, action_preds, return_preds = self._learn_model.forward(
            timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go
        )

        if self.cfg.env_type == 'atari':
            action_loss = F.cross_entropy(action_preds.reshape(-1, action_preds.size(-1)), action_target.reshape(-1))
        else:
            traj_mask = traj_mask.view(-1, )

            # only consider non padded elements
            action_preds = action_preds.view(-1, self.act_dim)[traj_mask > 0]

            if self._cfg.model.continuous:
                action_target = action_target.view(-1, self.act_dim)[traj_mask > 0]
                action_loss = F.mse_loss(action_preds, action_target)
            else:
                action_target = action_target.view(-1)[traj_mask > 0]
                action_loss = F.cross_entropy(action_preds, action_target)

        self._optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), 0.25)
        self._optimizer.step()
        self._scheduler.step()

        return {
            'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
            'action_loss': action_loss.detach().cpu().item(),
            'total_loss': action_loss.detach().cpu().item(),
        }

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = self._model
        # self._eval_model.reset()
        # init data
        self.device = torch.device(self._device)
        self.rtg_scale = self._cfg.rtg_scale  # normalize returns to go
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim
        self.eval_batch_size = self._cfg.evaluator_env_num
        self.max_eval_ep_len = self._cfg.max_eval_ep_len
        self.context_len = self._cfg.context_len  # K in decision transformer
        
        self.t = [0 for _ in range(self.eval_batch_size)]
        self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]
        self.timesteps = torch.arange(start=0, end=self.max_eval_ep_len, step=1).repeat(self.eval_batch_size, 1).to(self.device)
        if not self._cfg.model.continuous:
            self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.long, device=self.device)
        else:
            self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self.device)
        if self.cfg.env_type == 'atari':
            self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len,) + tuple(self.state_dim), dtype=torch.float32, device=self.device)
        else:
            self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self.device)
            self.state_mean = torch.from_numpy(self._cfg.state_mean).to(self.device)
            self.state_std = torch.from_numpy(self._cfg.state_std).to(self.device)
        self.rewards_to_go = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device)

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        # save and forward
        data_id = list(data.keys())
        data_len = len(data_id)
        
        self._eval_model.eval()
        with torch.no_grad():
            timesteps = torch.zeros((data_len, 1, 1), dtype=torch.long, device=self.device)
            if not self._cfg.model.continuous:
                actions = torch.zeros((data_len, self.context_len, 1), dtype=torch.long, device=self.device)
            else:
                actions = torch.zeros((data_len, self.context_len, self.act_dim), dtype=torch.float32, device=self.device)
            if self.cfg.env_type == 'atari':
                states = torch.zeros((data_len, self.context_len,) + tuple(self.state_dim), dtype=torch.float32, device=self.device)
            else:
                states = torch.zeros((data_len, self.context_len, self.state_dim), dtype=torch.float32, device=self.device)
            rewards_to_go = torch.zeros((data_len, self.context_len, 1), dtype=torch.float32, device=self.device)
            for i in data_id:
                if self.cfg.env_type == 'atari':
                    self.states[i, self.t[i]] = data[i]['obs'].to(self.device) / 255
                else:
                    self.states[i, self.t[i]] = (data[i]['obs'].to(self.device) - self.state_mean) / self.state_std
                # self.states[i, self.t[i]] = torch.tensor(data[i]['obs'])
                self.running_rtg[i] = self.running_rtg[i] - (data[i]['reward'].to(self.device) / self.rtg_scale)
                # self.running_rtg[i] = self.running_rtg[i] - (data[i]['reward'][0] / self.rtg_scale)
                self.rewards_to_go[i, self.t[i]] = self.running_rtg[i]
                
                if self.t[i] <= self.context_len:
                    if self.cfg.env_type == 'atari':
                        timesteps[i] = self.t[i] * torch.ones((1), dtype=torch.int64).to(self.device)
                    else:
                        timesteps[i] = self.timesteps[i, :self.context_len]
                    states[i] = self.states[i, :self.context_len]
                    actions[i] = self.actions[i, :self.context_len]
                    rewards_to_go[i] = self.rewards_to_go[i, :self.context_len]
                else:
                    if self.cfg.env_type == 'atari':
                        timesteps[i] = self.t[i] * torch.ones((1), dtype=torch.int64).to(self.device)
                    else:
                        timesteps[i] = self.timesteps[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    states[i] = self.states[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    actions[i] = self.actions[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    rewards_to_go[i] = self.rewards_to_go[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
            # if not self._cfg.model.continuous:
            #     actions = one_hot(actions.squeeze(-1), num=self.act_dim)
            _, act_preds, _ = self._eval_model.forward(timesteps, states, actions, rewards_to_go)
            del timesteps, states, actions, rewards_to_go
            act = torch.zeros((self.eval_batch_size, self.act_dim), dtype=torch.float32, device=self.device)
            for i in data_id:
                act[i] = act_preds[i, self.t[i]].detach() if self.t[i] < self.context_len else act_preds[i, -1].detach()
            if not self._cfg.model.continuous:
                act = torch.argmax(act, axis=1).unsqueeze(1)
            for i in data_id:
                self.actions[i, self.t[i]] = act[i]
                self.t[i] += 1
        if self._cuda:
            act = to_device(act, 'cpu')
        output = {'action': act}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: List[int] = None) -> None:
        # clean data
        if data_id is None:
            self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]
            self.t = [0 for _ in range(self.eval_batch_size)]
            self.timesteps = torch.arange(start=0, end=self.max_eval_ep_len, step=1).repeat(self.eval_batch_size, 1).to(self.device)
            if not self._cfg.model.continuous:
                self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.long, device=self.device)
            else:
                self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self.device)
            if self.cfg.env_type == 'atari':
                self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len,) + tuple(self.state_dim), dtype=torch.float32, device=self.device)
            else:
                self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self.device)
            self.rewards_to_go = torch.zeros((self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device)        
        else:
            for i in data_id:
                self.running_rtg[i] = self.rtg_target / self.rtg_scale
                self.t[i] = 0
                self.timesteps[i] = torch.arange(start=0, end=self.max_eval_ep_len, step=1).to(self.device)
                if not self._cfg.model.continuous:
                    self.actions[i] = torch.zeros((self.max_eval_ep_len, 1), dtype=torch.long, device=self.device)
                else:
                    self.actions[i] = torch.zeros((self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self.device)
                if self.cfg.env_type == 'atari':
                    self.states[i] = torch.zeros((self.max_eval_ep_len,) + tuple(self.state_dim), dtype=torch.float32, device=self.device)
                else:
                    self.states[i] = torch.zeros((self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self.device)
                self.rewards_to_go[i] = torch.zeros((self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device)

    def get_d4rl_normalized_score(self, score, env_name):
        env_key = env_name.split('-')[0].lower()
        assert env_key in D4RLTrajectoryDataset.REF_MAX_SCORE, \
            f'no reference score for {env_key} env to calculate d4rl score'
        d4rl_max_score, d4rl_min_score = D4RLTrajectoryDataset.REF_MAX_SCORE, D4RLTrajectoryDataset.REF_MIN_SCORE
        return (score - d4rl_min_score[env_key]) / (d4rl_max_score[env_key] - d4rl_min_score[env_key])

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            # 'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        # self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        
    def _load_state_dict_eval(self, state_dict: Dict[str, Any]) -> None:
        self._eval_model.load_state_dict(state_dict)
        # self._target_model.load_state_dict(state_dict['target_model'])
        # self._optimizer.load_state_dict(state_dict['optimizer'])

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'action_loss']

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        pass

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        pass