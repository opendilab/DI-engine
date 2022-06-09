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
from .dqn import DQNPolicy


@POLICY_REGISTRY.register('dt')
class DTPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of DT algorithm in discrete environments.
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
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
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

    def _init_learn(self) -> None:
        r"""
            Overview:
                Learn mode init method. Called by ``self.__init__``.
                Init the optimizer, algorithm config, main and target models.
            """

        self.stop_value = self._cfg.stop_value
        self.env_name = self._cfg.env_name
        dataset = self._cfg.dataset  # medium / medium-replay / medium-expert
        # rtg_scale: scale of `return to go`
        # rtg_target: max target of `return to go`
        # Our goal is normalize `return to go` to (0, 1), which will favour the covergence.
        # As a result, we usually set rtg_scale == rtg_target.
        self.rtg_scale = self._cfg.rtg_target  # normalize returns to go
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.max_eval_ep_len = self._cfg.max_eval_ep_len  # max len of one episode
        self.num_eval_ep = self._cfg.num_eval_ep  # num of evaluation episodes

        lr = self._cfg.learn.learning_rate  # learning rate
        wt_decay = self._cfg.wt_decay  # weight decay
        warmup_steps = self._cfg.warmup_steps  # warmup steps for lr scheduler

        max_train_iters = self._cfg.max_train_iters

        self.context_len = self._cfg.context_len  # K in decision transformer
        n_blocks = self._cfg.n_blocks  # num of transformer blocks
        embed_dim = self._cfg.embed_dim  # embedding (hidden) dim of transformer
        dropout_p = self._cfg.dropout_p  # dropout probability

        # # load data from this file
        # dataset_path = f'{self._cfg.dataset_dir}/{env_d4rl_name}.pkl'

        # saves model and csv in this directory
        self.log_dir = self._cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # training and evaluation device
        self.device = torch.device(self._device)

        self.start_time = datetime.now().replace(microsecond=0)
        self.start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")

        # prefix = "dt_" + env_d4rl_name
        self.prefix = "dt_" + self.env_name

        save_model_name = self.prefix + "_model_" + self.start_time_str + ".pt"
        self.save_model_path = os.path.join(self.log_dir, save_model_name)
        self.save_best_model_path = self.save_model_path[:-3] + "_best.pt"

        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        self.csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_header = (["duration", "num_updates", "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

        self.csv_writer.writerow(csv_header)

        dataset_path = self._cfg.learn.dataset_path
        logging.info("=" * 60)
        logging.info("start time: " + self.start_time_str)
        logging.info("=" * 60)

        logging.info("device set to: " + str(self.device))
        logging.info("dataset path: " + dataset_path)
        logging.info("model save path: " + self.save_model_path)
        logging.info("log csv save path: " + log_csv_path)

        self._env = gym.make(self.env_name)

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

        timesteps, states, actions, returns_to_go, traj_mask = data

        timesteps = timesteps.to(self.device)  # B x T
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device)  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T
        action_target = torch.clone(actions).detach().to(self.device)

        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        # if discrete
        if not self._cfg.model.continuous:
            actions = one_hot(actions.squeeze(-1), num=self.act_dim)

        state_preds, action_preds, return_preds = self._learn_model.forward(
            timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go
        )

        traj_mask = traj_mask.view(-1, )

        # only consider non padded elements
        action_preds = action_preds.view(-1, self.act_dim)[traj_mask > 0]

        if self._cfg.model.continuous:
            action_target = action_target.view(-1, self.act_dim)[traj_mask > 0]
        else:
            action_target = action_target.view(-1)[traj_mask > 0]

        if self._cfg.model.continuous:
            action_loss = F.mse_loss(action_preds, action_target)
        else:
            action_loss = F.cross_entropy(action_preds, action_target)

        self._optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), 0.25)
        self._optimizer.step()
        self._scheduler.step()

        return {
            'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
            'action_loss': action_loss.detach().cpu().item(),
        }

    def evaluate_on_env(self, state_mean=None, state_std=None, render=False):

        eval_batch_size = 1  # required for forward pass

        results = {}
        total_reward = 0
        total_timesteps = 0

        # state_dim = env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]

        if state_mean is None:
            self.state_mean = torch.zeros((self.state_dim, )).to(self.device)
        else:
            self.state_mean = torch.from_numpy(state_mean).to(self.device)

        if state_std is None:
            self.state_std = torch.ones((self.state_dim, )).to(self.device)
        else:
            self.state_std = torch.from_numpy(state_std).to(self.device)

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=self.max_eval_ep_len, step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        self._learn_model.eval()

        with torch.no_grad():

            for _ in range(self.num_eval_ep):

                # zeros place holders
                # continuous action
                actions = torch.zeros(
                    (eval_batch_size, self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self.device
                )

                # discrete action # TODO
                # actions = torch.randint(0,self.act_dim,[eval_batch_size, self.max_eval_ep_len, 1],
                # dtype=torch.long, device=self.device)

                states = torch.zeros(
                    (eval_batch_size, self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self.device
                )
                rewards_to_go = torch.zeros(
                    (eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self.device
                )

                # init episode
                running_state = self._env.reset()
                running_reward = 0
                running_rtg = self.rtg_target / self.rtg_scale

                for t in range(self.max_eval_ep_len):

                    total_timesteps += 1

                    # add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(running_state).to(self.device)
                    # states[0, t] = (states[0, t].cpu() - self.state_mean.cpu().numpy()) / self.state_std.cpu().numpy()
                    states[0, t] = (states[0, t] - self.state_mean) / self.state_std

                    # calcualate running rtg and add it in placeholder
                    running_rtg = running_rtg - (running_reward / self.rtg_scale)
                    rewards_to_go[0, t] = running_rtg

                    if t < self.context_len:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, :self.context_len], states[:, :self.context_len],
                            actions[:, :self.context_len], rewards_to_go[:, :self.context_len]
                        )
                        act = act_preds[0, t].detach()
                    else:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, t - self.context_len + 1:t + 1], states[:, t - self.context_len + 1:t + 1],
                            actions[:, t - self.context_len + 1:t + 1], rewards_to_go[:, t - self.context_len + 1:t + 1]
                        )
                        act = act_preds[0, -1].detach()

                    # if discrete
                    if not self._cfg.model.continuous:
                        act = torch.argmax(act)
                    running_state, running_reward, done, _ = self._env.step(act.cpu().numpy())

                    # add action in placeholder
                    actions[0, t] = act

                    total_reward += running_reward

                    if render:
                        self._env.render()
                    if done:
                        break

        results['eval/avg_reward'] = total_reward / self.num_eval_ep
        results['eval/avg_ep_len'] = total_timesteps / self.num_eval_ep

        return results

    def evaluate(self, total_update_times, state_mean=None, state_std=None, render=False):
        results = self.evaluate_on_env(state_mean, state_std, render)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = self.get_d4rl_normalized_score(results['eval/avg_reward'], self.env_name) * 100

        time_elapsed = str(datetime.now().replace(microsecond=0) - self.start_time)

        log_str = (
            "=" * 60 + '\n' + "time elapsed: " + time_elapsed + '\n' + "num of updates: " + str(total_update_times) +
            '\n' + '\n' + "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + "eval avg ep len: " +
            format(eval_avg_ep_len, ".5f") + '\n' + "eval d4rl score: " + format(eval_d4rl_score, ".5f")
        )

        logging.info(log_str)

        log_data = [time_elapsed, total_update_times, eval_avg_reward, eval_avg_ep_len, eval_d4rl_score]
        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        self.csv_writer.writerow(log_data)

        # save model
        logging.info("eval_avg_reward: " + format(eval_avg_reward, ".5f"))
        eval_env_score = eval_avg_reward
        if eval_env_score >= self.max_env_score:
            logging.info("saving max env score model at: " + self.save_best_model_path)
            torch.save(self._learn_model.state_dict(), self.save_best_model_path)
            self.max_env_score = eval_env_score

        logging.info("saving current model at: " + self.save_model_path)
        torch.save(self._learn_model.state_dict(), self.save_model_path)

        return self.max_env_score >= self.stop_value

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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dt', ['ding.model.template.decision_transformer']

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'action_loss']
