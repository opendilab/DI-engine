from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
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
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('cql')
class CQLPolicy(SACPolicy):
    r"""
       Overview:
           Policy class of CQL algorithm.

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
              |                                                |                                 | `\alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.repara_``   bool        True           | Determine whether to use        |
              | ``meterization``                               | reparameterization trick.       |
           12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | `\alpha`.                       | relative importance
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
        type='sac',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
        multi_agent=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,

            # (bool type) value_network: Determine whether to use value network as the
            # original SAC paper (arXiv 1801.01290).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            # value_network=False,

            # (str type) action_space: Use reparameterization trick for continous action
            action_space='reparameterization',

            # (int) Hidden size for actor network head.
            actor_head_hidden_size=256,

            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,

            # (float type) learning_rate_q: Learning rate for soft q network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_q=3e-4,
            # (float type) learning_rate_policy: Learning rate for policy network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_policy=3e-4,
            # (float type) learning_rate_value: Learning rate for value network.
            # `learning_rate_value` should be initialized, when model.value_network is True.
            # Please set to 3e-4, when model.value_network is True.
            learning_rate_value=3e-4,

            # (float type) learning_rate_alpha: Learning rate for auto temperature parameter `\alpha`.
            # Default to 3e-4.
            learning_rate_alpha=3e-4,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,

            # (float type) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,

            # (bool type) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
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
            # (int) The numbers of action sample each at every state s from a uniform-at-random
            num_actions=10,
            # (bool) Whether use lagrange multiplier in q value loss.
            with_lagrange=False,
            # (float) The threshold for difference in Q-values
            lagrange_thresh=-1,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
            # (bool) Whether to use entropy in target q.
            with_q_entropy=False,
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
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_network = False
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
        self._model.actor[2].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].mu.bias.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.bias.data.uniform_(-init_w, init_w)
        if self._twin_critic:
            self._model.critic[0][2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic[0][2].last.bias.data.uniform_(-init_w, init_w)
            self._model.critic[1][2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic[1][2].last.bias.data.uniform_(-init_w, init_w)
        else:
            self._model.critic[2].last.weight.data.uniform_(-init_w, init_w)
            self._model.critic[2].last.bias.data.uniform_(-init_w, init_w)

        # Optimizers
        if self._value_network:
            self._optimizer_value = Adam(
                self._model.value_critic.parameters(),
                lr=self._cfg.learn.learning_rate_value,
            )
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
            self._target_entropy = self._cfg.learn.get('target_entropy', -np.prod(self._cfg.model.action_shape))
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

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
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
        if self._value_network:
            # predict v value
            v_value = self._learn_model.forward(obs, mode='compute_value_critic')['v_value']
            with torch.no_grad():
                next_v_value = self._target_model.forward(next_obs, mode='compute_value_critic')['v_value']
            target_q_value = next_v_value
        else:
            # target q value.
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

        # 7. (optional)compute value loss
        if self._value_network:
            # new_q_value: (bs, ), log_prob: (bs, act_shape) -> target_v_value: (bs, )
            if self._with_q_entropy:
                target_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
            else:
                target_v_value = new_q_value.unsqueeze(-1).mean(dim=-1)
            loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

            # update value network
            self._optimizer_value.zero_grad()
            loss_dict['value_loss'].backward()
            self._optimizer_value.step()

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

    def _get_policy_actions(self, data: Dict, num_actions=10, epsilon: float = 1e-6) -> List:

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

    def _get_q_value(self, data: Dict, keep=True) -> torch.Tensor:
        new_q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = [value.view(-1, self._num_actions, 1) for value in new_q_value]
        else:
            new_q_value = new_q_value.view(-1, self._num_actions, 1)
        if self._twin_critic and not keep:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        return new_q_value


@POLICY_REGISTRY.register('cql_discrete')
class CQLDiscretePolicy(DQNPolicy):
    """
        Overview:
            Policy class of CQL algorithm in discrete environments.

        Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qrdqn          | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     True           | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        6  | ``other.eps``      float    0.05           | Start value for epsilon decay. It's
           | ``.start``                                 | small because rainbow use noisy net.
        7  | ``other.eps``      float    0.05           | End value for epsilon decay.
           | ``.end``
        8  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        9  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        10 | ``learn.update``   int      3              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        11 ``learn.kappa``      float    /              | Threshold of Huber loss
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='cql_discrete',
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
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
        ),
        # collect_mode config
        collect=dict(
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, )
        ),
    )

    def _init_learn(self) -> None:
        r"""
            Overview:
                Learn mode init method. Called by ``self.__init__``.
                Init the optimizer, algorithm config, main and target models.
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

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
            Overview:
                Forward and backward function of learn mode.
            Arguments:
                - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
            Returns:
                - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
            """
        data = default_preprocess_learn(
            data, use_priority=self._priority, ignore_done=self._cfg.learn.ignore_done, use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
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
        if self._cfg.learn.multi_gpu:
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

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of collect mode(collect training data), with eps_greedy for exploration.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting policy_output(action) for the interaction with \
                env and the constructing of transition.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
            Overview:
                Get the trajectory and the n step return data, then sample from the n_step return data
            Arguments:
                - data (:obj:`list`): The trajectory's cache
            Returns:
                - samples (:obj:`dict`): The training samples generated
            """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qrdqn', ['ding.model.template.q_learning']

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss', 'q_target', 'q_value']
