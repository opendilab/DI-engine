from typing import List, Dict, Any, Tuple, Union
import copy
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


@POLICY_REGISTRY.register('iql')
class IQLPolicy(Policy):
    """
    Overview:
        Policy class of Implicit Q-Learning (IQL) algorithm for continuous control. Paper link: https://arxiv.org/abs/2110.06169.

    Config:
        == ====================  ========    =============  ================================= =======================
        ID Symbol                Type        Default Value  Description                       Other(Shape)
        == ====================  ========    =============  ================================= =======================
        1  ``type``              str         iql            | RL policy register name, refer  | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY`` | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network |
        3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
           | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
           |                                                | buffer when training starts.    | TD3.
        4  | ``model.policy_``   int         256            | Linear layer size for policy    |
           | ``embedding_size``                             | network.                        |
        5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
           | ``embedding_size``                             | network.                        |
        6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
           | ``embedding_size``                             | network.                        | model.value_network
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
        10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
           |                                                | coefficient.                    | zation for auto
           |                                                |                                 | `alpha`, when
           |                                                |                                 | auto_alpha is True
        11 | ``learn.repara_``   bool        True           | Determine whether to use        |
           | ``meterization``                               | reparameterization trick.       |
        12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
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
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='iql',
        # (bool) Whether to use cuda for policy.
        cuda=False,
        # (bool) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        on_policy=False,
        # (bool) priority: Determine whether to use priority in buffer sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        random_collect_size=10000,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,
            # (str type) action_space: Use reparameterization trick for continous action
            action_space='reparameterization',
            # (int) Hidden size for actor network head.
            actor_head_hidden_size=512,
            actor_head_layer_num=3,
            # (int) Hidden size for critic network head.
            critic_head_hidden_size=512,
            critic_head_layer_num=2,
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) learning_rate_q: Learning rate for soft q network.
            learning_rate_q=3e-4,
            # (float) learning_rate_policy: Learning rate for policy network.
            learning_rate_policy=3e-4,
            # (float) learning_rate_alpha: Learning rate for auto temperature parameter ``alpha``.
            learning_rate_alpha=3e-4,
            # (float) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (float) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,
            # (bool) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            auto_alpha=True,
            # (bool) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer.
            init_w=3e-3,
            # (int) The numbers of action sample each at every state s from a uniform-at-random.
            num_actions=10,
            # (bool) Whether use lagrange multiplier in q value loss.
            with_lagrange=False,
            # (float) The threshold for difference in Q-values.
            lagrange_thresh=-1,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
            # (float) coefficient for the asymmetric loss, range from [0.5, 1.0], default to 0.70.
            tau=0.7,
            # (float) temperature coefficient for Advantage Weighted Regression loss, default to 1.0.
            beta=1.0,
        ),
        eval=dict(),  # for compatibility
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """

        return 'continuous_qvac', ['ding.model.template.qvac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For SAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma, min_q_weight, with_lagrange, \
            main and target model. Especially, the ``auto_alpha`` mechanism for balancing max entropy \
            target is also initialized here.
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
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._twin_critic = self._cfg.model.twin_critic
        self._num_actions = self._cfg.learn.num_actions

        self._min_q_version = 3
        self._min_q_weight = self._cfg.learn.min_q_weight
        self._with_lagrange = self._cfg.learn.with_lagrange and (self._lagrange_thresh > 0)
        self._lagrange_thresh = self._cfg.learn.lagrange_thresh
        if self._with_lagrange:
            self.target_action_gap = self._lagrange_thresh
            self.log_alpha_prime = torch.tensor(0.).to(self._device).requires_grad_()
            self.alpha_prime_optimizer = Adam(
                [self.log_alpha_prime],
                lr=self._cfg.learn.learning_rate_q,
            )

        # Weight Init
        init_w = self._cfg.learn.init_w
        self._model.actor_head[-1].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].mu.bias.data.uniform_(-init_w, init_w)
        # self._model.actor_head[-1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        # self._model.actor_head[-1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)
        if self._twin_critic:
            self._model.critic_q_head[0][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_q_head[0][-1].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic_q_head[1][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_q_head[1][-1].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[0][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[0][-1].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[1][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[1][-1].last.bias.data.uniform_(-init_w, init_w)
        else:
            self._model.critic_q_head[2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_q_head[-1].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_v_head[-1].last.bias.data.uniform_(-init_w, init_w)

        # Optimizers
        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

        self._forward_learn_cnt = 0

        self._tau = self._cfg.learn.tau
        self._beta = self._cfg.learn.beta
        self._policy_start_training_counter = 10000  #300000

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the offline dataset and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For IQL, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight``.
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
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']

        # 1. predict q and v value
        value = self._learn_model.forward(data, mode='compute_critic')
        q_value, v_value = value['q_value'], value['v_value']

        # 2. predict target value
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

            next_obs_dist = TransformedDistribution(
                Independent(Normal(mu, sigma), 1),
                transforms=[TanhTransform(cache_size=1),
                            AffineTransform(loc=0.0, scale=1.05)]
            )
            next_action = next_obs_dist.rsample()
            next_log_prob = next_obs_dist.log_prob(next_action)

            next_data = {'obs': next_obs, 'action': next_action}
            next_value = self._learn_model.forward(next_data, mode='compute_critic')
            next_q_value, next_v_value = next_value['q_value'], next_value['v_value']

            # the value of a policy according to the maximum entropy objective
            if self._twin_critic:
                next_q_value = torch.min(next_q_value[0], next_q_value[1])

        # 3. compute v loss
        if self._twin_critic:
            q_value_min = torch.min(q_value[0], q_value[1]).detach()
            v_loss_0 = asymmetric_l2_loss(q_value_min - v_value[0], self._tau)
            v_loss_1 = asymmetric_l2_loss(q_value_min - v_value[1], self._tau)
            v_loss = (v_loss_0 + v_loss_1) / 2
        else:
            advantage = q_value.detach() - v_value
            v_loss = asymmetric_l2_loss(advantage, self._tau)

        # 4. compute q loss
        if self._twin_critic:
            next_v_value = torch.min(next_v_value[0], next_v_value[1])
            q_data0 = v_1step_td_data(q_value[0], next_v_value, reward, done, data['weight'])
            loss_dict['critic_q_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], next_v_value, reward, done, data['weight'])
            loss_dict['twin_critic_q_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            q_loss = (loss_dict['critic_q_loss'] + loss_dict['twin_critic_q_loss']) / 2
        else:
            q_data = v_1step_td_data(q_value, next_v_value, reward, done, data['weight'])
            loss_dict['critic_q_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)
            q_loss = loss_dict['critic_q_loss']

        # 5. update q and v network
        self._optimizer_q.zero_grad()
        v_loss.backward()
        q_loss.backward()
        self._optimizer_q.step()

        # 6. evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']

        dist = TransformedDistribution(
            Independent(Normal(mu, sigma), 1),
            transforms=[TanhTransform(cache_size=1), AffineTransform(loc=0.0, scale=1.05)]
        )
        action = data['action']
        log_prob = dist.log_prob(action)

        eval_data = {'obs': obs, 'action': action}
        new_value = self._learn_model.forward(eval_data, mode='compute_critic')
        new_q_value, new_v_value = new_value['q_value'], new_value['v_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
            new_v_value = torch.min(new_v_value[0], new_v_value[1])
        new_advantage = new_q_value - new_v_value

        # 8. compute policy loss
        policy_loss = (-log_prob * torch.exp(new_advantage.detach() / self._beta).clamp(max=20.0)).mean()
        self._policy_start_training_counter -= 1

        loss_dict['policy_loss'] = policy_loss

        # 9. update policy network
        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self._model.actor.parameters(), 1)
        self._optimizer_policy.step()

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1

        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': q_loss.abs().tolist(),
            'q_loss': q_loss.detach().mean().item(),
            'v_loss': v_loss.detach().mean().item(),
            'log_prob': log_prob.detach().mean().item(),
            'next_q_value': next_q_value.detach().mean().item(),
            'next_v_value': next_v_value.detach().mean().item(),
            'policy_loss': policy_loss.detach().mean().item(),
            'total_loss': loss_dict['total_loss'].detach().item(),
            'advantage_max': new_advantage.max().detach().item(),
            'new_q_value': new_q_value.detach().mean().item(),
            'new_v_value': new_v_value.detach().mean().item(),
            'policy_grad_norm': policy_grad_norm,
        }

    def _get_policy_actions(self, data: Dict, num_actions: int = 10, epsilon: float = 1e-6) -> List:
        # evaluate to get action distribution
        obs = data['obs']
        obs = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)

        # evaluate action log prob depending on Jacobi determinant.
        y = 1 - action.pow(2) + epsilon
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        return action, log_prob.view(-1, num_actions, 1)

    def _get_q_value(self, data: Dict, keep: bool = True) -> torch.Tensor:
        new_q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = [value.view(-1, self._num_actions, 1) for value in new_q_value]
        else:
            new_q_value = new_q_value.view(-1, self._num_actions, 1)
        if self._twin_critic and not keep:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        return new_q_value

    def _get_v_value(self, data: Dict, keep: bool = True) -> torch.Tensor:
        new_v_value = self._learn_model.forward(data, mode='compute_critic')['v_value']
        if self._twin_critic:
            new_v_value = [value.view(-1, self._num_actions, 1) for value in new_v_value]
        else:
            new_v_value = new_v_value.view(-1, self._num_actions, 1)
        if self._twin_critic and not keep:
            new_v_value = torch.min(new_v_value[0], new_v_value[1])
        return new_v_value

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For SAC, it contains the \
            collect_model other algorithm-specific arguments such as unroll_len. \
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._collect_model.forward(data, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            action = torch.tanh(dist.rsample())
            output = {'logit': (mu, sigma), 'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For continuous SAC, it contains obs, next_obs, action, reward, done. The logit \
            will be also added when ``collector_logit`` is True.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For continuous SAC, it contains the action and the logit (mu and sigma) of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        if self._cfg.collect.collector_logit:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'logit': policy_output['logit'],
                'action': policy_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        else:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': policy_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In continuous SAC, a train sample is a processed transition \
            (unroll_len=1).
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training.
        """
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For SAC, it contains the \
            eval model, which is equipped with ``base`` model wrapper to ensure compability.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu) / 1.05  # deterministic_eval
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        return [
            'cur_lr_q',
            'cur_lr_p',
            'value_loss'
            'policy_loss',
            'q_loss',
            'v_loss',
            'policy_loss',
            'log_prob',
            'total_loss',
            'advantage_max',
            'next_q_value',
            'next_v_value',
            'new_q_value',
            'new_v_value',
            'policy_grad_norm',
        ] + twin_critic

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
