from typing import List, Dict, Any, Tuple, Optional
from collections import namedtuple
import torch.nn.functional as F
import torch
import numpy as np
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_decollate
from ding.policy.dt import DTPolicy
from ding.model import model_wrap

@POLICY_REGISTRY.register('promptdt')
class PDTPolicy(DTPolicy):
    """
    Overview:
        Policy class of Decision Transformer algorithm in discrete environments.
        Paper link: https://arxiv.org/pdf/2206.13499.
    """
    def default_model(self) -> Tuple[str, List[str]]:
        return 'dt', ['ding.model.template.decision_transformer']
    
    def _init_learn(self) -> None:
        super()._init_learn()
        self.need_prompt = self._cfg.need_prompt

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
        self.have_train = True

        if self._cuda:
            data = to_device(data, self._device)

        p_s, p_a, p_rtg, p_t, p_mask, timesteps, states, actions, rewards, returns_to_go, \
            traj_mask = [], [], [], [], [], [], [], [], [], [], []

        for d in data:
            if self.need_prompt:
                p, timestep, s, a, r, rtg, mask = d
                timesteps.append(timestep)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                returns_to_go.append(rtg)
                traj_mask.append(mask)
                ps, pa, prtg, pt, pm = p
                p_s.append(ps)
                p_a.append(pa)
                p_rtg.append(prtg)
                p_mask.append(pm)
                p_t.append(pt)
            else:
                timestep, s, a, r, rtg, mask = d
                timesteps.append(timestep)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                returns_to_go.append(rtg)
                traj_mask.append(mask)
        
        timesteps = torch.stack(timesteps, dim=0)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        returns_to_go = torch.stack(returns_to_go, dim=0)
        traj_mask = torch.stack(traj_mask, dim=0)
        if self.need_prompt:
            p_s = torch.stack(p_s, dim=0)
            p_a = torch.stack(p_a, dim=0)
            p_rtg = torch.stack(p_rtg, dim=0)
            p_mask = torch.stack(p_mask, dim=0)
            p_t = torch.stack(p_t, dim=0)
            prompt = (p_s, p_a, p_rtg, p_t, p_mask)
        else:
            prompt = None

        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        state_preds, action_preds, return_preds = self._learn_model.forward(
            timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go, prompt=prompt
        )

        traj_mask = traj_mask.view(-1, )

        # only consider non padded elements
        action_preds = action_preds.reshape(-1, self.act_dim)[traj_mask > 0]

        action_target = actions.reshape(-1, self.act_dim)[traj_mask > 0]
        action_loss = F.mse_loss(action_preds, action_target)

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
    
    def init_dataprocess_func(self, dataloader):
        self.dataloader = dataloader
    
    def _init_eval(self) -> None:
        self.test_num = self._cfg.learn.test_num
        self._eval_model = self._model
        self.eval_batch_size = self._cfg.evaluator_env_num
        self.rtg_target = self._cfg.rtg_target
        self.task_id = None
        self.test_task_id = [[] for _ in range(self.eval_batch_size)]
        self.have_train =False
        if self._cfg.model.continuous:
            self.actions = torch.zeros(
                (self.eval_batch_size, self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self._device
            )
        else:
            self.actions = torch.zeros(
                (self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.long, device=self._device
            )

        self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]
        self.states = torch.zeros(
            (self.eval_batch_size, self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self._device
        )
        self.timesteps = torch.arange(
            start=0, end=self.max_eval_ep_len, step=1
        ).repeat(self.eval_batch_size, 1).to(self._device)
        self.rewards_to_go = torch.zeros(
            (self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self._device
        )

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        if self.need_prompt:
            p_s, p_a, p_rtg, p_t, p_mask = [], [], [], [], []
            for i in range(self.eval_batch_size):
                ps, pa, prtg, pt, pm = self.dataloader.get_prompt(is_test=True, id=self.task_id[i])
                p_s.append(ps)
                p_a.append(pa)
                p_rtg.append(prtg)
                p_mask.append(pm)
                p_t.append(pt)
            p_s = torch.stack(p_s, dim=0).to(self._device)
            p_a = torch.stack(p_a, dim=0).to(self._device)
            p_rtg = torch.stack(p_rtg, dim=0).to(self._device)
            p_mask = torch.stack(p_mask, dim=0).to(self._device)
            p_t = torch.stack(p_t, dim=0).to(self._device)
            prompt = (p_s, p_a, p_rtg, p_t, p_mask)
        else:
            prompt = None
            
        data_id = list(data.keys())

        self._eval_model.eval()
        with torch.no_grad():
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
                self.states[i, self.t[i]] = self.dataloader.normalize(data[i]['obs'], 'obs', self.task_id[i])
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
                act = torch.argmax(logits, axis=1).unsqueeze(1)
            else:
                act = logits
            for i in data_id:
                self.actions[i, self.t[i]] = act[i]  # TODO: self.actions[i] should be a queue when exceed max_t
                self.t[i] += 1

        if self._cuda:
            act = to_device(act, 'cpu')
        output = {'action': act}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def warm_train(self, id: int):
        self.task_id = [id] * self.eval_batch_size
    
    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        if self.have_train:
            if self.task_id is None:
                self.task_id = [0] * self.eval_batch_size

        if data_id is None:
            self.t = [0 for _ in range(self.eval_batch_size)]
            self.timesteps = torch.arange(
                start=0, end=self.max_eval_ep_len, step=1
            ).repeat(self.eval_batch_size, 1).to(self._device)
            if not self._cfg.model.continuous:
                self.actions = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.long, device=self._device
                )
            else:
                self.actions = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len, self.act_dim),
                    dtype=torch.float32,
                    device=self._device
                )
            if self._atari_env:
                self.states = torch.zeros(
                    (
                        self.eval_batch_size,
                        self.max_eval_ep_len,
                    ) + tuple(self.state_dim),
                    dtype=torch.float32,
                    device=self._device
                )
                self.running_rtg = [self.rtg_target for _ in range(self.eval_batch_size)]
            else:
                self.states = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len, self.state_dim),
                    dtype=torch.float32,
                    device=self._device
                )
                self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]

            self.rewards_to_go = torch.zeros(
                (self.eval_batch_size, self.max_eval_ep_len, 1), dtype=torch.float32, device=self._device
            )
        else:
            for i in data_id:
                self.t[i] = 0
                if not self._cfg.model.continuous:
                    self.actions[i] = torch.zeros((self.max_eval_ep_len, 1), dtype=torch.long, device=self._device)
                else:
                    self.actions[i] = torch.zeros(
                        (self.max_eval_ep_len, self.act_dim), dtype=torch.float32, device=self._device
                    )
                if self._atari_env:
                    self.states[i] = torch.zeros(
                        (self.max_eval_ep_len, ) + tuple(self.state_dim), dtype=torch.float32, device=self._device
                    )
                    self.running_rtg[i] = self.rtg_target
                else:
                    self.states[i] = torch.zeros(
                        (self.max_eval_ep_len, self.state_dim), dtype=torch.float32, device=self._device
                    )
                    self.running_rtg[i] = self.rtg_target / self.rtg_scale
                    self.timesteps[i] = torch.arange(start=0, end=self.max_eval_ep_len, step=1).to(self._device)
                self.rewards_to_go[i] = torch.zeros((self.max_eval_ep_len, 1), dtype=torch.float32, device=self._device)
