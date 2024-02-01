from typing import List, Dict, Any, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .sac import SACPolicy
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('edac')
class EDACPolicy(SACPolicy):
    """
       Overview:
           Policy class of EDAC algorithm. Paper link: https://arxiv.org/pdf/2110.01548.pdf

       Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other(Shape)
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         td3            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           4  | ``model.policy_``   int         256            | Linear layer size for policy    |
              | ``embedding_size``                             | network.                        |
           5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
              | ``embedding_size``                             | network.                        |
           6  | ``model.emsemble``  int         10             | Number of Q-ensemble network    |
              | ``_num``                                       |                                 |
              |                                                |                                 | is False.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
              | ``_rate_q``                                    | network.                        | model.value_network
              |                                                |                                 | is True.
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
              | ``_rate_policy``                               | network.                        | model.value_network
              |                                                |                                 | is True.
           9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
              | ``_rate_value``                                | network.                        | model.value_network
              |                                                |                                 | is False.
           10 | ``learn.alpha``     float       1.0            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | `alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.eta``       bool        True           | Parameter of EDAC algorithm     | Defalut to 1.0
           12 | ``learn.``          bool        True           | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | `alpha`.                        | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in halfcheetah env.
           14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
    """
    config = dict(
        # (str) RL policy register name
        type='edac',
        cuda=False,
        on_policy=False,
        multi_agent=False,
        priority=False,
        priority_IS_weight=False,
        random_collect_size=10000,
        model=dict(
            # (bool type) ensemble_num:num of Q-network.
            ensemble_num=10,
            # (bool type) value_network: Determine whether to use value network as the
            # original EDAC paper (arXiv 2110.01548).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            # value_network=False,

            # (int) Hidden size for actor network head.
            actor_head_hidden_size=256,

            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_value=3e-4,
            learning_rate_alpha=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=1,
            auto_alpha=True,
            # (bool type) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
            # (bool) Whether to use entropy in target q.
            with_q_entropy=False,
            eta=0.1,
        ),
        collect=dict(
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=1000000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'edac', ['ding.model.template.edac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For EDAC, in addition \
            to the things that need to be initialized in SAC, it is also necessary to additionally define \
            eta/with_q_entropy/forward_learn_cnt. \
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
        super()._init_learn()
        # EDAC special implementation
        self._eta = self._cfg.learn.eta
        self._with_q_entropy = self._cfg.learn.with_q_entropy
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: List[Dict[int, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For EDAC, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``logit``, ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys like ``weight``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for EDACPolicy: \
            ``ding.policy.tests.test_edac``.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if len(data.get('action').shape) == 1:
            data['action'] = data['action'].reshape(-1, 1)

        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        acs = data['action']

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            next_action = torch.tanh(pred)
            y = 1 - next_action.pow(2) + 1e-6
            next_log_prob = dist.log_prob(pred).unsqueeze(-1)
            next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective

            target_q_value, _ = torch.min(target_q_value, dim=0)
            if self._with_q_entropy:
                target_q_value -= self._alpha * next_log_prob.squeeze(-1)
            target_q_value = self._gamma * (1 - done) * target_q_value + reward

        weight = data['weight']
        if weight is None:
            weight = torch.ones_like(q_value)
        td_error_per_sample = nn.MSELoss(reduction='none')(q_value, target_q_value).mean(dim=1).sum()
        loss_dict['critic_loss'] = (td_error_per_sample * weight).mean()

        # penalty term of EDAC
        if self._eta > 0:
            # [batch_size,dim] -> [Ensemble_num,batch_size,dim]
            pre_obs = obs.unsqueeze(0).repeat_interleave(self._cfg.model.ensemble_num, dim=0)
            pre_acs = acs.unsqueeze(0).repeat_interleave(self._cfg.model.ensemble_num, dim=0).requires_grad_(True)

            # [Ensemble_num,batch_size]
            q_pred_tile = self._learn_model.forward({
                'obs': pre_obs,
                'action': pre_acs
            }, mode='compute_critic')['q_value'].requires_grad_(True)

            q_pred_grads = torch.autograd.grad(q_pred_tile.sum(), pre_acs, retain_graph=True, create_graph=True)[0]
            q_pred_grads = q_pred_grads / (torch.norm(q_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            # [Ensemble_num,batch_size,act_dim] -> [batch_size,Ensemble_num,act_dim]
            q_pred_grads = q_pred_grads.transpose(0, 1)

            q_pred_grads = q_pred_grads @ q_pred_grads.permute(0, 2, 1)
            masks = torch.eye(
                self._cfg.model.ensemble_num, device=obs.device
            ).unsqueeze(dim=0).repeat(q_pred_grads.size(0), 1, 1)
            q_pred_grads = (1 - masks) * q_pred_grads
            grad_loss = torch.mean(torch.sum(q_pred_grads, dim=(1, 2))) / (self._cfg.model.ensemble_num - 1)
            loss_dict['critic_loss'] += grad_loss * self._eta

        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_q.step()

        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        new_q_value, _ = torch.min(new_q_value, dim=0)

        # 8. compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # 9. update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # 10. compute alpha loss
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }
