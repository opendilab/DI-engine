from typing import List, Dict, Any, Tuple, Optional
from collections import namedtuple
import torch.nn.functional as F
import torch
import numpy as np
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_decollate
from ding.policy.dt import DTPolicy

@POLICY_REGISTRY.register('promptdt')
class PDTPolicy(DTPolicy):
    """
    Overview:
        Policy class of Decision Transformer algorithm in discrete environments.
        Paper link: https://arxiv.org/pdf/2206.13499.
    """
    def _forward_learn(self, data: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the offline dataset and then returns the output \
            result, including various training information such as loss, current learning rate.
        Arguments:
            - data (:obj:`List[torch.Tensor]`): The input data used for policy forward, including a series of \
                processed torch.Tensor data, i.e., timesteps, states, actions, returns_to_go, traj_mask.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        """
        self._learn_model.train()

        prompt, timesteps, states, actions, rewards, returns_to_go, traj_mask = data

        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        if self._basic_discrete_env:
            actions = actions.to(torch.long)
            actions = actions.squeeze(-1)
            action_target = torch.clone(actions).detach().to(self._device)

        if self._atari_env:
            state_preds, action_preds, return_preds = self._learn_model.forward(
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go, tar=1, prompt=prompt
            )
        else:
            state_preds, action_preds, return_preds = self._learn_model.forward(
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go, prompt=prompt
            )

        if self._atari_env:
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
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), self.clip_grad_norm_p)
        self._optimizer.step()
        self._scheduler.step()

        return {
            'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
            'action_loss': action_loss.detach().cpu().item(),
            'total_loss': action_loss.detach().cpu().item(),
        }
    
    def get_dataloader(self, dataloader):
        self.dataloader = dataloader
    
    def _init_eval(self) -> None:
        self.task_id = [0] * self.eval_batch_size
        super()._init_eval()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        prompt = []
        for i in range(self.eval_batch_size):
            prompt.append(self.dataloader.get_prompt(is_test=True, id=self.task_id[i]))
            
        prompt = torch.tensor(prompt, device=self._device)

        data_id = list(data.keys())

        self._eval_model.eval()
        with torch.no_grad():
            if self._atari_env:
                states = torch.zeros(
                    (
                        self.eval_batch_size,
                        self.context_len,
                    ) + tuple(self.state_dim),
                    dtype=torch.float32,
                    device=self._device
                )
                timesteps = torch.zeros((self.eval_batch_size, 1, 1), dtype=torch.long, device=self._device)
            else:
                states = torch.zeros(
                    (self.eval_batch_size, self.context_len, self.state_dim), dtype=torch.float32, device=self._device
                )
                timesteps = torch.zeros((self.eval_batch_size, self.context_len), dtype=torch.long, device=self._device)
            if not self._cfg.model.continuous:
                actions = torch.zeros(
                    (self.eval_batch_size, self.context_len, 1), dtype=torch.long, device=self._device
                )
            else:
                actions = torch.zeros(
                    (self.eval_batch_size, self.context_len, self.act_dim), dtype=torch.float32, device=self._device
                )
            rewards_to_go = torch.zeros(
                (self.eval_batch_size, self.context_len, 1), dtype=torch.float32, device=self._device
            )
            for i in data_id:
                if self._atari_env:
                    self.states[i, self.t[i]] = data[i]['obs'].to(self._device)
                else:
                    self.states[i, self.t[i]] = (data[i]['obs'].to(self._device) - self.state_mean) / self.state_std
                self.running_rtg[i] = self.running_rtg[i] - data[i]['reward'].to(self._device)
                self.rewards_to_go[i, self.t[i]] = self.running_rtg[i]

                if self.t[i] <= self.context_len:
                    if self._atari_env:
                        timesteps[i] = min(self.t[i], self._cfg.model.max_timestep) * torch.ones(
                            (1, 1), dtype=torch.int64
                        ).to(self._device)
                    else:
                        timesteps[i] = self.timesteps[i, :self.context_len]
                    states[i] = self.states[i, :self.context_len]
                    actions[i] = self.actions[i, :self.context_len]
                    rewards_to_go[i] = self.rewards_to_go[i, :self.context_len]
                else:
                    if self._atari_env:
                        timesteps[i] = min(self.t[i], self._cfg.model.max_timestep) * torch.ones(
                            (1, 1), dtype=torch.int64
                        ).to(self._device)
                    else:
                        timesteps[i] = self.timesteps[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    states[i] = self.states[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    actions[i] = self.actions[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
                    rewards_to_go[i] = self.rewards_to_go[i, self.t[i] - self.context_len + 1:self.t[i] + 1]
            if self._basic_discrete_env:
                actions = actions.squeeze(-1)

            _, act_preds, _ = self._eval_model.forward(timesteps, states, actions, rewards_to_go, prompt=prompt)
            del timesteps, states, actions, rewards_to_go

            logits = act_preds[:, -1, :]
            if not self._cfg.model.continuous:
                if self._atari_env:
                    probs = F.softmax(logits, dim=-1)
                    act = torch.zeros((self.eval_batch_size, 1), dtype=torch.long, device=self._device)
                    for i in data_id:
                        act[i] = torch.multinomial(probs[i], num_samples=1)
                else:
                    act = torch.argmax(logits, axis=1).unsqueeze(1)
            for i in data_id:
                self.actions[i, self.t[i]] = act[i]  # TODO: self.actions[i] should be a queue when exceed max_t
                self.t[i] += 1

        if self._cuda:
            act = to_device(act, 'cpu')
        output = {'action': act}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    
    
    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self.task_id[data_id] += 1
        super()._reset_eval(data_id)