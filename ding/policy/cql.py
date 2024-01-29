from typing import List, Dict, Any, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .sac import SACPolicy
from .qrdqn import QRDQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('cql')
class CQLPolicy(SACPolicy):
    """
    Overview:
        Policy class of CQL algorithm for continuous control. Paper link: https://arxiv.org/abs/2006.04779.

    Config:
        == ====================  ========    =============  ================================= =======================
        ID Symbol                Type        Default Value  Description                       Other(Shape)
        == ====================  ========    =============  ================================= =======================
        1  ``type``              str         cql            | RL policy register name, refer  | this arg is optional,
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
        type='cql',
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
            actor_head_hidden_size=256,
            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
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
            # (bool) Whether to use entropy in target q.
            with_q_entropy=False,
        ),
        eval=dict(),  # for compatibility
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For SAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma, min_q_weight, with_lagrange and \
            with_q_entropy, main and target model. Especially, the ``auto_alpha`` mechanism for balancing max entropy \
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

        self._with_q_entropy = self._cfg.learn.with_q_entropy

        # Weight Init
        init_w = self._cfg.learn.init_w
        self._model.actor_head[-1].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].mu.bias.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)
        if self._twin_critic:
            self._model.critic_head[0][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_head[0][-1].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic_head[1][-1].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_head[1][-1].last.bias.data.uniform_(-init_w, init_w)
        else:
            self._model.critic_head[2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic_head[-1].last.bias.data.uniform_(-init_w, init_w)

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
        # Init auto alpha
        if self._cfg.learn.auto_alpha:
            if self._cfg.learn.target_entropy is None:
                assert 'action_shape' in self._cfg.model, "CQL need network model with action_shape variable"
                self._target_entropy = -np.prod(self._cfg.model.action_shape)
            else:
                self._target_entropy = self._cfg.learn.target_entropy
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
                assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = torch.FloatTensor([self._cfg.learn.alpha]).to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._alpha], lr=self._cfg.learn.learning_rate_alpha)
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha], requires_grad=False, device=self._device, dtype=torch.float32
            )
            self._auto_alpha = False

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

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
                For CQL, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
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
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # 2. predict target value
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
            if self._twin_critic:
                # find min one as target q value
                if self._with_q_entropy:
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
                else:
                    target_q_value = torch.min(target_q_value[0], target_q_value[1])
            else:
                if self._with_q_entropy:
                    target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 4. add CQL

        curr_actions_tensor, curr_log_pis = self._get_policy_actions(data, self._num_actions)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions({'obs': next_obs}, self._num_actions)

        random_actions_tensor = torch.FloatTensor(curr_actions_tensor.shape).uniform_(-1,
                                                                                      1).to(curr_actions_tensor.device)

        obs_repeat = obs.unsqueeze(1).repeat(1, self._num_actions,
                                             1).view(obs.shape[0] * self._num_actions, obs.shape[1])
        act_repeat = data['action'].unsqueeze(1).repeat(1, self._num_actions, 1).view(
            data['action'].shape[0] * self._num_actions, data['action'].shape[1]
        )

        q_rand = self._get_q_value({'obs': obs_repeat, 'action': random_actions_tensor})
        # q2_rand = self._get_q_value(obs, random_actions_tensor, network=self.qf2)
        q_curr_actions = self._get_q_value({'obs': obs_repeat, 'action': curr_actions_tensor})
        # q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q_next_actions = self._get_q_value({'obs': obs_repeat, 'action': new_curr_actions_tensor})
        # q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        cat_q1 = torch.cat([q_rand[0], q_value[0].reshape(-1, 1, 1), q_next_actions[0], q_curr_actions[0]], 1)
        cat_q2 = torch.cat([q_rand[1], q_value[1].reshape(-1, 1, 1), q_next_actions[1], q_curr_actions[1]], 1)
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
        if self._min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [
                    q_rand[0] - random_density, q_next_actions[0] - new_log_pis.detach(),
                    q_curr_actions[0] - curr_log_pis.detach()
                ], 1
            )
            cat_q2 = torch.cat(
                [
                    q_rand[1] - random_density, q_next_actions[1] - new_log_pis.detach(),
                    q_curr_actions[1] - curr_log_pis.detach()
                ], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1, dim=1).mean() * self._min_q_weight
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1).mean() * self._min_q_weight
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q_value[0].mean() * self._min_q_weight
        min_qf2_loss = min_qf2_loss - q_value[1].mean() * self._min_q_weight

        if self._with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        loss_dict['critic_loss'] += min_qf1_loss
        if self._twin_critic:
            loss_dict['twin_critic_loss'] += min_qf2_loss

        # 5. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward(retain_graph=True)
        if self._twin_critic:
            loss_dict['twin_critic_loss'].backward()
        self._optimizer_q.step()

        # 6. evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

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


@POLICY_REGISTRY.register('discrete_cql')
class DiscreteCQLPolicy(QRDQNPolicy):
    """
    Overview:
        Policy class of discrete CQL algorithm in discrete action space environments.
        Paper link: https://arxiv.org/abs/2006.04779.
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='discrete_cql',
        # (bool) Whether to use cuda for policy.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for one gradient descent.
            batch_size=64,
            # (float) Learning rate for soft q network.
            learning_rate=0.001,
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env).
            ignore_done=False,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
        ),
        eval=dict(),  # for compatibility
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For DiscreteCQL, it mainly \
            contains the optimizer, algorithm-specific arguments such as gamma, nstep and min_q_weight, main and \
            target model. This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._min_q_weight = self._cfg.learn.min_q_weight
        self._priority = self._cfg.priority
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use wrapper instead of plugin
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

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
                For DiscreteCQL, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys like ``weight`` \
                and ``value_gamma`` for nstep return computation.
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
        data = default_preprocess_learn(
            data, use_priority=self._priority, ignore_done=self._cfg.learn.ignore_done, use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        if data['action'].dim() == 2 and data['action'].shape[-1] == 1:
            data['action'] = data['action'].squeeze(-1)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        ret = self._learn_model.forward(data['obs'])
        q_value, tau = ret['q'], ret['tau']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['q']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        # add CQL
        # 1. chose action and compute q in dataset.
        # 2. compute value loss(negative_sampling - dataset_expec)
        replay_action_one_hot = F.one_hot(data['action'], self._cfg.model.action_shape)
        replay_chosen_q = (q_value.mean(-1) * replay_action_one_hot).sum(dim=1)

        dataset_expec = replay_chosen_q.mean()

        negative_sampling = torch.logsumexp(q_value.mean(-1), dim=1).mean()

        min_q_loss = negative_sampling - dataset_expec

        data_n = qrdqn_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], tau, data['weight']
        )
        value_gamma = data.get('value_gamma')
        loss, td_error_per_sample = qrdqn_nstep_td_error(
            data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
        )

        loss += self._min_q_weight * min_q_loss

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            'q_target': target_q_value.mean().item(),
            'q_value': q_value.mean().item(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'total_loss', 'q_target', 'q_value']
