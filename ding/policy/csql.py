from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('csql')
class ContinousSoftQLPolicy(Policy):
    r"""
       Overview:
           Policy class of SAC algorithm.

           https://arxiv.org/pdf/1801.01290.pdf

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
        type='softql',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
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
            twin_critic=False,

            # (bool type) value_network: Determine whether to use value network as the
            # original SAC paper (arXiv 1801.01290).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            # value_network=False,
            actor_head_type='noise_regression',
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
            auto_alpha=False,
            # (bool type) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
            sample_size=100,
            kernel_update_ratio=0.5
        ),
        collect=dict(
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            noise_sigma=0.1
        ),
        eval=dict(
            evaluator=dict(
                # (int) Evaluate every "eval_freq" training iterations.
                eval_freq=5000,
            ),
        ),
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

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_network = False  # TODO self._cfg.model.value_network
        self._twin_critic = self._cfg.model.twin_critic

        # Weight Init for the last output layer
        # init_w = self._cfg.learn.init_w
        # self._model.actor[2].mu.weight.data.uniform_(-init_w, init_w)
        # self._model.actor[2].mu.bias.data.uniform_(-init_w, init_w)
        # self._model.actor[2].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        # self._model.actor[2].log_sigma_layer.bias.data.uniform_(-init_w, init_w)

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
        self._sample_size = self._cfg.learn.sample_size
        self._kernel_update_ratio = self._cfg.learn.kernel_update_ratio

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
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr, loss, target_q_value and other \
                running information.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        obs_size = obs.shape[1]
        batch_size = obs.shape[0]
        action_size = self._model.action_shape
        sample_size = self._sample_size
        kernel_update_ratio = self._kernel_update_ratio
        alpha = self._alpha
        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        # 2. predict target next value
        # equation (10)
        with torch.no_grad():
            obs_expand = next_obs.unsqueeze(1)
            obs_expand = obs_expand.expand([batch_size, sample_size, obs_size])
            obs_expand = obs_expand.reshape(batch_size * sample_size, obs_size)
            action_expand = torch.rand(batch_size * sample_size, action_size)
            sample_data = {'obs': obs_expand, 'action': action_expand}
            next_q_expand = self._target_model.forward(sample_data, mode='compute_critic')['q_value']
            next_q_expand = next_q_expand.reshape(batch_size, -1)
            # next_q_expand : batch_size x sample_size
            next_v_value = alpha * torch.logsumexp(next_q_expand / alpha, 1)
            # Importance weights add just a constant to the value.
            next_v_value -= torch.log(torch.tensor(sample_size))
            next_v_value += torch.tensor(action_size).type(torch.FloatTensor) * torch.log(torch.tensor(2))
        # 3. compute td error
        q_data = v_1step_td_data(q_value, next_v_value, reward, done, data['weight'])
        loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # create svgd update
        # mu = torch.zeros(action_size)
        # sigma = torch.ones(action_size)
        # dist = Independent(Normal(mu, sigma), 1)
        # sample_list = dist.sample_n(batch_size * sample_size)
        # sample_list = torch.randn(batch_size * sample_size)
        sample_list = torch.randn((batch_size * sample_size, action_size))
        latent_sample = sample_list.view(batch_size, sample_size, -1)
        # latent sample shape: [batch_size, sample_size, action_size]
        obs_expand = obs.unsqueeze(1)
        obs_expand = obs_expand.expand(batch_size, sample_size, obs_size)
        stochastic_expand = torch.cat([obs_expand, latent_sample], dim=2)
        stochastic_expand = stochastic_expand.reshape(batch_size * sample_size, -1)
        # stochastic_expand shape: [batch_size * sample_size, obs_shape + action_sample]
        sample_action_flat = self._learn_model.forward(stochastic_expand, mode='compute_actor')['action']
        sample_action = sample_action_flat.reshape(batch_size, sample_size, action_size)
        n_updated_actions = int(kernel_update_ratio * sample_size)
        n_fixed_actions = sample_size - n_updated_actions
        fixed_actions, updated_actions = torch.split(sample_action, [n_fixed_actions, n_updated_actions], dim=1)
        # fixed actions shape: batch_size, n_fixed_actions, action_size
        # updated actions shape: batch_size, n_updated_actions, action_size
        #fixed_actions = fixed_actions.detach()
        fixed_observations, updated_observations = torch.split(obs_expand, [n_fixed_actions, n_updated_actions], dim=1)
        fixed_actions_flat = fixed_actions.reshape(-1, action_size)
        fixed_observations_flat = fixed_observations.reshape(-1, obs_size)
        # fixed_actions_flat shape:     batch_size*n_fixed_actions x action_size
        # fixed_observation_flat shape: batch_size*n_fixed_actions x obs_size
        data_fixed = {'obs': fixed_observations_flat, 'action': fixed_actions_flat}
        svgd_target_values = self._target_model.forward(data_fixed, mode='compute_critic')['q_value']

        # Here we must check if tanh is in the nn.parameters or not
        squash_correction = (1 - fixed_actions_flat.pow(2) + 1e-6).sum(-1)
        log_p = svgd_target_values + squash_correction
        grad_logp = torch.autograd.grad(outputs=log_p, inputs=fixed_actions, grad_outputs=torch.ones_like(log_p))[0]
        grad_log_p = grad_logp.unsqueeze(2)
        grad_log_p = grad_log_p.detach()
        # grad_log_p shape : batch x n_fixed_actions x 1 x action_dim, how to get
        #assert_shape(grad_log_p, [batch_size, n_fixed_actions, 1, action_size])
        kernel_dict = self.adaptive_isotropic_gaussian_kernel(xs=fixed_actions, ys=updated_actions)
        # kernel_dict kappa output shape: batch_size x n_fixed_acitons x n_updated_shape
        # kernel dict kappa gradient shape: batch_size x n_fixed_actions x n_updated_shape x aciton_size
        #kernel function in Equation 13:
        kappa = kernel_dict['output'].unsqueeze(3)
        # kappa shape: batch_size x n_fixed_actions x n_updated_actions x 1
        # grad_log_p : batch_size x n_fixed_actions x 1 x action_size
        #assert_shape(kappa, [batch_size, n_fixed_actions, n_updated_actions, 1])
        # Stein Variational Gradient in Equation 13:
        action_gradients = (kappa * grad_log_p + kernel_dict['gradient']).mean(1)
        # to batch_size x n_updated_actions x action_size
        #assert_shape(action_gradients, [batch_size, n_updated_actions, action_size])

        surrogate_loss = torch.sum(action_gradients.mul(updated_actions))
        loss_dict['policy_loss'] = surrogate_loss

        # update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()
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
            #'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }

    def adaptive_isotropic_gaussian_kernel(self, xs, ys, h_min=1e-3):
        # fixed actions: batch_size, fixed_action_size, action_size
        # updated actions: batch_size, updated_action_size, action_size
        _, Kx, D = xs.shape
        _, Ky, D2 = ys.shape
        assert D == D2

        leading_shape = xs.shape[:-2]
        Kx, D = xs.shape[-2], xs.shape[-1]
        Ky, D2 = ys.shape[-2], ys.shape[-1]
        assert D == D2

        #input shape: batch_size x fixed_action_size * updated_action_size
        input_shape = leading_shape + torch.Size([Kx * Ky])

        # Compute the pairwise distances of the left and right particles
        diff = torch.unsqueeze(xs, -2) - torch.unsqueeze(ys, -3)
        # diff shape: batch_size x fixed_action_size x updated_action_size x action_size
        dist_sq = torch.sum(diff ** 2, axis=-1, keepdim=False)
        # dist_sq shape: batch_size x fixed_action_size x updated_action_size
        values, _ = torch.topk(input=torch.reshape(dist_sq, input_shape), k=(Kx * Ky // 2 + 1))
        # median shape: batch
        median_sq = values[..., -1]

        h = median_sq / np.log(Kx)
        #h = torch.max(h, h_min)
        h[h < h_min] = h_min
        h_nograd = h.detach()
        h_expanded_twice = torch.unsqueeze(torch.unsqueeze(h_nograd, -1), -1)
        # h_expanded_twice shape: batch_size x 1 x 1

        kappa = torch.exp(-dist_sq / h_expanded_twice)
        # kappa shape: batch_size x fixed_action_size x updated_action_size

        h_expanded_thrice = torch.unsqueeze(h_expanded_twice, -1)
        # h_expanded_thrice shape: batch_size x 1 x 1 x 1
        kappa_expanded = torch.unsqueeze(kappa, -1)
        # kappa_expanded shape : batch_size x fixed_action_size x updated_action_size x 1
        kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
        # kappa_grad shape: batch_size x fixed_aciton_size x updated_action_size x action_size
        return {"output": kappa, "gradient": kappa_grad}

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._value_network:
            ret.update({'optimizer_value': self._optimizer_value.state_dict()})
        if self._auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        if self._value_network:
            self._optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if self._auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            action_size = 3
            # mu = torch.zeros(action_size)
            # sigma = torch.ones(action_size)
            # dist = Independent(Normal(mu, sigma), 1)
            # sample_list = dist.sample(data.shape[0])
            sample_list = torch.randn((data.shape[0], action_size))
            stochastic_list = torch.cat([data, sample_list], dim=1)
            stochastic_data = {'obs': stochastic_list}
            sample_action = self._collect_model.forward(stochastic_list, mode='compute_actor')
            output = sample_action
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - policy_output (:obj:`dict`): Output of policy collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            action_size = 3
            # mu = torch.zeros(action_size)
            # sigma = torch.ones(action_size)
            # dist = Independent(Normal(mu, sigma), 1)
            # sample_list = dist.sample(data.shape[0])
            sample_list = torch.randn((data.shape[0], action_size))
            stochastic_list = torch.cat([data, sample_list], dim=1)
            #stochastic_data = {'obs': stochastic_list}
            sample_action = self._collect_model.forward(stochastic_list, mode='compute_actor')
            #output = {'action': sample_action}
            output = sample_action
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac', ['ding.model.template.qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        value_loss = ['value_loss'] if self._value_network else []
        return [
            'alpha_loss',
            'policy_loss',
            'critic_loss',
            'cur_lr_q',
            'cur_lr_p',
            'target_q_value',
            'alpha',
            'td_error',
        ] + twin_critic + alpha_loss + value_loss
