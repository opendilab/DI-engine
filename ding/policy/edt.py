from typing import List, Dict, Any, Tuple, Optional
import math
from collections import namedtuple
import torch.nn.functional as F
import torch
import numpy as np
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_decollate
from .base_policy import Policy

REF_MIN_SCORE = {
    'halfcheetah' : -280.178953,
    'walker2d' : 1.629008,
    'hopper' : -20.272305,
    'ant' : -325.6,
    'antmaze' : 0.0,
}
 
REF_MAX_SCORE = {
    'halfcheetah' : 12135.0,
    'walker2d' : 4592.3,
    'hopper' : 3234.3,
    'ant' : 3879.7,
    'antmaze' : 700,
}

@POLICY_REGISTRY.register('edt')
class EDTPolicy(Policy):
    """
    Overview:
        Policy class of Decision Transformer algorithm in discrete environments.
        Paper link: https://arxiv.org/abs/2106.01345.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='edt',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (int) N-step reward for target q_value estimation
        obs_shape=4,
        action_shape=2,
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        batch_size=64,  # training batch size
        wt_decay=1e-4,  # decay weight in optimizer
        warmup_steps=10000,  # steps for learning rate warmup
        context_len=20,  # length of transformer input
        learning_rate=1e-4,
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about DQN, its registered name is ``dqn`` and the import_names is \
            ``ding.model.template.q_learning``.
        """
        return 'edt', ['ding.model.template.edt']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For Decision Transformer, \
            it mainly contains the optimizer, algorithm-specific arguments such as rtg_scale and lr scheduler.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        # rtg_scale: scale of `return to go`
        # rtg_target: max target of `return to go`
        # Our goal is normalize `return to go` to (0, 1), which will favour the covergence.
        # As a result, we usually set rtg_scale == rtg_target.
        self.env_name = self._cfg.env_id
        
        self.rtg_scale = self._cfg.rtg_scale  # normalize returns to go
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.max_eval_ep_len = self._cfg.max_eval_ep_len  # max len of one episode
        
        self.expectile = self._cfg.weights.expectile
        self.top_percentile = self._cfg.weights.top_percentile
        self.expert_weight = self._cfg.weights.expert_weight
        self.exp_loss_weight = self._cfg.weights.exp_loss_weight
        self.state_loss_weight = self._cfg.weights.state_loss_weight
        self.cross_entropy_weight = self._cfg.weights.cross_entropy_weight
        
        
        
        lr = self._cfg.learning_rate  # learning rate
        wt_decay = self._cfg.wt_decay  # weight decay
        warmup_steps = self._cfg.warmup_steps  # warmup steps for lr scheduler

        self.clip_grad_norm_p = self._cfg.clip_grad_norm_p
        
        self.context_len = self._cfg.model.context_len  # K in decision transformer
        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim
        self.num_bin = self._cfg.model.num_bin # num of bin 

        self._learn_model = self._model
        self._atari_env = 'state_mean' not in self._cfg
        self._basic_discrete_env = not self._cfg.model.continuous and 'state_mean' in self._cfg

        if self._atari_env:
            self._optimizer = self._learn_model.configure_optimizers(wt_decay, lr)
        else:
            self._optimizer = torch.optim.AdamW(self._learn_model.parameters(), lr=lr, weight_decay=wt_decay)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        self.max_env_score = -1.0

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

        timesteps, states, next_states, actions, returns_to_go, rewards, traj_mask = data

        # The shape of `returns_to_go` may differ with different dataset (B x T or B x T x 1),
        # and we need a 3-dim tensor
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        if len(rewards.shape) == 2:
            rewards = rewards.unsqueeze(-1)
        # Guarantee return and reward has shape [B, T, 1]

        if self._basic_discrete_env:
            actions = actions.to(torch.long)
            actions = actions.squeeze(-1)
        action_target = torch.clone(actions).detach().to(self._device) # [B, T, A]
        state_target = torch.clone(states).detach().to(self._device) # [B, T, S]
        return_to_go_target = torch.clone(returns_to_go).detach().to(self._device)

        if self._atari_env:
            state_preds, action_preds, return_preds, imp_return_preds, reward_preds = self._learn_model.forward(
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go, tar=1
            )
        else:
            state_preds, action_preds, return_preds, imp_return_preds, reward_preds = self._learn_model.forward(
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go
            )
        def expectile_loss(diff: torch.Tensor, expectile: float=0.8) -> torch.Tensor:
            weight = torch.where(diff > 0, expectile, (1 - expectile))
            return weight * (diff**2)

        def cross_entropy(logits, labels):
            # labels = F.one_hot(labels.long(), num_classes=int(num_bin)).squeeze(2)
            labels = F.one_hot(
                labels.long(), num_classes=int(self.num_bin)
            ).squeeze()
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(logits, labels.float())
        
        def encode_return(env_name, ret, scale=1.0, num_bin=120, rtg_scale=1000):
            env_key = env_name.split("-")[0].lower()
            if env_key not in REF_MAX_SCORE:
                ret_max = 100
            else:
                ret_max = REF_MAX_SCORE[env_key]
            if env_key not in REF_MIN_SCORE:
                ret_min = -20
            else:
                ret_min = REF_MIN_SCORE[env_key]
            ret_max /= rtg_scale
            ret_min /= rtg_scale
            interval = (ret_max - ret_min) / (num_bin-1)
            ret = torch.clip(ret, ret_min, ret_max)
            return ((ret - ret_min) // interval).float()
        
        
        
        if self._atari_env:
            action_loss = F.cross_entropy(action_preds.reshape(-1, action_preds.size(-1)), action_target.reshape(-1))
        else:
            traj_mask = traj_mask.view(-1, )

            # only consider non padded elements
            action_preds = action_preds.view(-1, self.act_dim)[traj_mask > 0]
            state_preds = state_preds.view(-1, self.state_dim)[traj_mask > 0]
            imp_return_preds = imp_return_preds.reshape(-1, 1)[traj_mask > 0]
            return_preds = return_preds.reshape(-1, int(self.num_bin))[traj_mask > 0]
            
            
            if self._cfg.model.continuous:
                action_target = action_target.view(-1, self.act_dim)[traj_mask > 0]
                action_loss = F.mse_loss(action_preds, action_target)
                state_target = next_states.view(-1, self.state_dim)[traj_mask > 0]
                state_loss = F.mse_loss(state_preds, state_target)
                imp_return_target = returns_to_go.reshape(-1, 1)[traj_mask > 0]
                imp_loss = expectile_loss((imp_return_target - imp_return_preds), self.expectile).mean()
                return_target = (
                    encode_return(
                        self.env_name,
                        returns_to_go,
                        num_bin=self.num_bin,
                        rtg_scale=self.rtg_scale,
                    ).float().reshape(-1, 1)[traj_mask > 0]
                )
                return_cross_entropy_loss = cross_entropy(return_preds, return_target)
                
            else:
                action_target = action_target.view(-1)[traj_mask > 0]
                action_loss = F.cross_entropy(action_preds, action_target)
                state_target = next_states.view(-1)[traj_mask > 0]
                state_loss = F.cross_entropy(state_preds, state_target)
                imp_return_target = returns_to_go.reshape(-1, 1)[traj_mask > 0]
                imp_loss = expectile_loss((imp_return_target - imp_return_preds), self.expectile).mean()
                
        edt_loss = action_loss \
            + state_loss * self.state_loss_weight \
            + imp_loss * self.exp_loss_weight \
        
        if self._cfg.model.continuous:
            edt_loss += return_cross_entropy_loss * self.cross_entropy_weight
        
        self._optimizer.zero_grad()
        edt_loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), self.clip_grad_norm_p)
        self._optimizer.step()
        self._scheduler.step()

        return {
            'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
            'action_loss': action_loss.detach().cpu().item(),
            'state_loss': state_loss.detach().cpu().item(),
            'implict_loss': imp_loss.detach().cpu().item(),
            'total_loss': edt_loss.detach().cpu().item(),
        }

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For DQN, it contains the \
            eval model, some algorithm-specific parameters such as context_len, max_eval_ep_len, etc.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. tip::
            For the evaluation of complete episodes, we need to maintain some historical information for transformer \
            inference. These variables need to be initialized in ``_init_eval`` and reset in ``_reset_eval`` when \
            necessary.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = self._model
        # init data
        self._device = torch.device(self._device)
        
        self.real_rtg = self._cfg.real_rtg
        self.rtg_scale = self._cfg.rtg_scale  # normalize returns to go
        self.rtg_target = self._cfg.rtg_target  # max target reward_to_go
        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim
        self.eval_batch_size = self._cfg.evaluator_env_num
        self.max_eval_ep_len = self._cfg.max_eval_ep_len
        self.context_len = self._cfg.model.context_len  # K in decision transformer
        self.expectile = self._cfg.weights.expectile
        
        self.rs_steps = self._cfg.eval.rs_steps
        self.rs_ratio = self._cfg.weights.rs_ratio
        self.heuristic = self._cfg.eval.heuristic
        self.heuristic_delta = self._cfg.eval.heuristic_delta
        

        
        self.t = [0 for _ in range(self.eval_batch_size)]
        if self._cfg.model.continuous:
            self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, self.act_dim), 
                                    dtype=torch.float32, device=self._device)
        else:
            # (B, eval_len + 2 * context_len, A) for actions
            self.actions = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, 1), 
                                       dtype=torch.long, device=self._device)
        
        self._atari_env = 'state_mean' not in self._cfg
        self._basic_discrete_env = not self._cfg.model.continuous and 'state_mean' in self._cfg
        
        if self._atari_env:
            self.running_rtg = [self.rtg_target for _ in range(self.eval_batch_size)]
            self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len,) + tuple(self.state_dim),
                                    dtype=torch.float32, device=self._device)
        else:
            # (B, eval_len + 2 * context_len, S) for states
            self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]
            self.states = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, self.state_dim), 
                                    dtype=torch.float32, device=self._device)
            self.state_mean = torch.from_numpy(np.array(self._cfg.state_mean)).to(self._device)
            self.state_std = torch.from_numpy(np.array(self._cfg.state_std)).to(self._device)
        
        
        self.timesteps = torch.arange(start=0, end=self.max_eval_ep_len + 2 * self.context_len, step=1)
        self.timesteps = self.timesteps.repeat(self.eval_batch_size, 1).to(self._device)
        
        # (B, eval_len + 2 * context_len, 1) for rtg & rewards
        self.rewards_to_go = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, 1), 
                                        dtype=torch.float32, device=self._device)
        self.rewards = torch.zeros((self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, 1), 
                                        dtype=torch.float32, device=self._device)
    
    def decode_return(env_name: str, ret, scale: float=1.0, num_bin: int=120, rtg_scale: int=1000):
            env_key = env_name.split("-")[0].lower()
            if env_key not in REF_MAX_SCORE:
                ret_max = 100
            else:
                ret_max = REF_MAX_SCORE[env_key]
            if env_key not in REF_MIN_SCORE:
                ret_min = -20
            else:
                ret_min = REF_MIN_SCORE[env_key]
            ret_max /= rtg_scale
            ret_min /= rtg_scale
            interval = (ret_max - ret_min) / num_bin
            return ret * interval + ret_min
        
    def _return_heuristic(self,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards_to_go: torch.Tensor,
        rewards: torch.Tensor,
        context_len: int, 
        t: int,
        # top_percentile: float,
        # num_bin: int,
        # rtg_scale: int, 
        # expert_weight: float,
        # mgdt_sampling: bool = False,
        rs_steps: int = 2,
        rs_ratio: int = 1,
        real_rtg: bool = False,
        use_heuristic: bool = False,
        heuristic_delta: int = 1,
        previous_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        highest_ret = -9999
        estimated_rtg = None
        best_i = 0
        best_act = None
        if t < context_len:
            for i in range(0, math.ceil((t + 1) / rs_ratio), rs_steps):
                _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    rewards_to_go[:, i : context_len + i],
                    rewards[:, i : context_len + i],
                )
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    imp_ret_preds,
                    rewards[:, i : context_len + i],
                )
                if not real_rtg:
                    imp_ret_preds = imp_ret_preds_pure
                ret_i = imp_ret_preds[:, t - i].detach().item()
                if ret_i > highest_ret:
                    highest_ret = ret_i
                    best_i = i
                    estimated_rtg = imp_ret_preds.detach()
                    best_act = act_preds[0, t - i].detach()
        else:
            if use_heuristic:
                prev_best_index = context_len - previous_index
                loop = (prev_best_index-heuristic_delta, prev_best_index+1+heuristic_delta)
            else:
                loop = (0, math.ceil(context_len/rs_ratio), rs_steps)
            for i in range(*loop):
                if use_heuristic and (i < 0 or i >= context_len):
                    continue
                _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    rewards_to_go[:, t - context_len + 1 + i : t + 1 + i],
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    imp_ret_preds,
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )
                if not real_rtg:
                    imp_ret_preds = imp_ret_preds_pure
                
                ret_i = imp_ret_preds[:, -1 - i].detach().item()
                if ret_i > highest_ret:
                    highest_ret = ret_i
                    best_i = i
                    # estimated_rtg = imp_ret_preds.detach()
                    best_act = act_preds[0, -1 - i].detach()
        return best_act, context_len - best_i
    
    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance, such as interacting with envs. \
            Forward means that the policy gets some input data (current obs/return-to-go and historical information) \
            from the envs and then returns the output data, such as the action to interact with the envs. \
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs and \
                reward to calculate running return-to-go. The key of the dict is environment id and the value is the \
                corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            Decision Transformer will do different operations for different types of envs in evaluation.
        """
        # save and forward
        
        
        data_id = list(data.keys())

        self._eval_model.eval()
        with torch.no_grad():            
            
            print(self.t)
            best_acts = []
            for i in data_id:
                curr_states = self.states[i].unsqueeze(0)
                curr_runninng_rtg = self.running_rtg[i]
                curr_rewards_to_go = self.rewards_to_go[i].unsqueeze(0)
                curr_rewards = self.rewards[i].unsqueeze(0)
                curr_actions = self.actions[i].unsqueeze(0)
                previous_index = None
                for t in range(self.max_eval_ep_len):
                    if self._atari_env:
                        curr_states[0, t] = data[i]['obs'].to(self._device)
                    else:
                        curr_states[0, t] = (data[i]['obs'].to(self._device) - self.state_mean) / self.state_std
                        # print(f"curr_states[0, t] 的 shape 是 {curr_states[0, t].shape}, 而 state的shape是{curr_states.shape}")
                    curr_runninng_rtg = curr_runninng_rtg - (data[i]['reward'] / self.rtg_scale).to(self._device)
                    curr_rewards_to_go[0, t] = curr_runninng_rtg
                    curr_rewards[0, t] = data[i]['reward']
                    act, best_index = self._return_heuristic(
                    model=self._eval_model,
                    timesteps=self.timesteps[i].unsqueeze(0),
                    states=curr_states,
                    actions=curr_actions,
                    rewards_to_go=curr_rewards_to_go,
                    rewards=curr_rewards,
                    context_len=self.context_len,
                    t=t,
                    rs_steps=self.rs_steps,
                    rs_ratio=self.rs_ratio,
                    real_rtg=self.real_rtg,
                    use_heuristic=self.heuristic,
                    heuristic_delta=self.heuristic_delta,
                    previous_index=previous_index
                )
                previous_index = best_index
                best_acts.append(act)
            acts = torch.stack(best_acts, dim=0)
            print(f"acts has shape {acts.shape}")
            # previous_index = None
            # for t in range(self.max_eval_ep_len):
            #     if self._atari_env:
            #         self.states[0, t] = data[0]['obs'].to(self._device)
            #     else:
            #         self.states[0, t] = (data[0]['obs'].to(self._device) - self.state_mean) / self.state_std
            #         print(f"self.states[0, t] 的 shape 是 {self.states[0, t].shape}, 而 state的shape是{self.states.shape}")
            #     self.running_rtg[0] = self.running_rtg[0] - (data[0]['reward'] / self.rtg_scale).to(self._device)
            #     self.rewards_to_go[0, t] = self.running_rtg[0]
            #     self.rewards[0, t] = data[0]['reward']
            #     act, best_index = self._return_heuristic(
            #         model=self._eval_model,
            #         timesteps=self.timesteps,
            #         states=self.states,
            #         actions=self.actions,
            #         rewards_to_go=self.rewards_to_go,
            #         rewards=self.rewards,
            #         context_len=self.context_len,
            #         t=t,
            #         rs_steps=self.rs_steps,
            #         rs_ratio=self.rs_ratio,
            #         real_rtg=self.real_rtg,
            #         use_heuristic=self.heuristic,
            #         heuristic_delta=self.heuristic_delta,
            #         previous_index=previous_index
            #     )
            #     previous_index = best_index
            #     act = act.unsqueeze(0)
            #     # print(f"{t} ended! act has shape {act.shape}")
            for i in data_id:
                self.actions[i, self.t[i]] = acts[i]  # TODO: self.actions[i] should be a queue when exceed max_t
                self.t[i] += 1   
            
            if self._cuda:
                acts = to_device(acts, 'cpu')
            output = {'action': acts}
            output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset some statvaeful riables for eval mode when necessary, such as the historical info of transformer \
            for decision transformer. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different environments/episodes in evaluation in ``data_id`` will have different history.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                specified by ``data_id``.
        """
        # clean data
        if data_id is None:
            self.t = [0 for _ in range(self.eval_batch_size)]
            self.timesteps = torch.arange(
                start=0, end=self.max_eval_ep_len + 2 * self.context_len, step=1
            ).repeat(self.eval_batch_size, 1).to(self._device)
            if not self._cfg.model.continuous:
                self.actions = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, 1), dtype=torch.long, device=self._device
                )
            else:
                self.actions = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, self.act_dim),
                    dtype=torch.float32,
                    device=self._device
                )
            if self._atari_env:
                self.states = torch.zeros(
                    (
                        self.eval_batch_size,
                        self.max_eval_ep_len + 2 * self.context_len,
                    ) + tuple(self.state_dim),
                    dtype=torch.float32,
                    device=self._device
                )
                self.running_rtg = [self.rtg_target for _ in range(self.eval_batch_size)]
            else:
                self.states = torch.zeros(
                    (self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, self.state_dim),
                    dtype=torch.float32,
                    device=self._device
                )
                self.running_rtg = [self.rtg_target / self.rtg_scale for _ in range(self.eval_batch_size)]

            self.rewards_to_go = torch.zeros(
                (self.eval_batch_size, self.max_eval_ep_len + 2 * self.context_len, 1), dtype=torch.float32, device=self._device
            )
        else:
            for i in data_id:
                self.t[i] = 0
                if not self._cfg.model.continuous:
                    self.actions[i] = torch.zeros((self.max_eval_ep_len + 2 * self.context_len, 1), dtype=torch.long, device=self._device)
                else:
                    self.actions[i] = torch.zeros(
                        (self.max_eval_ep_len + 2 * self.context_len, self.act_dim), dtype=torch.float32, device=self._device
                    )
                if self._atari_env:
                    self.states[i] = torch.zeros(
                        (self.max_eval_ep_len + 2 * self.context_len, ) + tuple(self.state_dim), dtype=torch.float32, device=self._device
                    )
                    self.running_rtg[i] = self.rtg_target
                else:
                    self.states[i] = torch.zeros(
                        (self.max_eval_ep_len + 2 * self.context_len, self.state_dim), dtype=torch.float32, device=self._device
                    )
                    self.running_rtg[i] = self.rtg_target / self.rtg_scale
                    self.timesteps[i] = torch.arange(start=0, end=self.max_eval_ep_len + 2 * self.context_len, step=1).to(self._device)
                self.rewards_to_go[i] = torch.zeros((self.max_eval_ep_len + 2 * self.context_len, 1), dtype=torch.float32, device=self._device)
                self.rewards[i] = torch.zeros((self.max_eval_ep_len + 2 * self.context_len, 1), dtype=torch.float32, device=self._device)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'action_loss']

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        pass

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        pass
