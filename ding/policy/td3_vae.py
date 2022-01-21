from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from .ddpg import DDPGPolicy
from ding.model.template.vae import VanillaVAE
from ding.utils import RunningMeanStd
from torch.nn import functional as F


@POLICY_REGISTRY.register('td3-vae')
class TD3VAEPolicy(DDPGPolicy):
    r"""
    Overview:
        Policy class of TD3 algorithm.

        Since DDPG and TD3 share many common things, we can easily derive this TD3
        class from DDPG class by changing ``_actor_update_freq``, ``_twin_critic`` and noise in model wrapper.

        https://arxiv.org/pdf/1802.09477.pdf

    Property:
        learn_mode, collect_mode, eval_mode

    Config:

    == ====================  ========    ==================  =================================   =======================
    ID Symbol                Type        Default Value       Description                         Other(Shape)
    == ====================  ========    ==================  =================================   =======================
    1  ``type``              str         td3                 | RL policy register name, refer    | this arg is optional,
                                                             | to registry ``POLICY_REGISTRY``   | a placeholder
    2  ``cuda``              bool        True                | Whether to use cuda for network   |
    3  | ``random_``         int         25000               | Number of randomly collected      | Default to 25000 for
       | ``collect_size``                                    | training samples in replay        | DDPG/TD3, 10000 for
       |                                                     | buffer when training starts.      | sac.
    4  | ``model.twin_``     bool        True                | Whether to use two critic         | Default True for TD3,
       | ``critic``                                          | networks or only one.             | Clipped Double
       |                                                     |                                   | Q-learning method in
       |                                                     |                                   | TD3 paper.
    5  | ``learn.learning``  float       1e-3                | Learning rate for actor           |
       | ``_rate_actor``                                     | network(aka. policy).             |
    6  | ``learn.learning``  float       1e-3                | Learning rates for critic         |
       | ``_rate_critic``                                    | network (aka. Q-network).         |
    7  | ``learn.actor_``    int         2                   | When critic network updates       | Default 2 for TD3, 1
       | ``update_freq``                                     | once, how many times will actor   | for DDPG. Delayed
       |                                                     | network update.                   | Policy Updates method
       |                                                     |                                   | in TD3 paper.
    8  | ``learn.noise``     bool        True                | Whether to add noise on target    | Default True for TD3,
       |                                                     | network's action.                 | False for DDPG.
       |                                                     |                                   | Target Policy Smoo-
       |                                                     |                                   | thing Regularization
       |                                                     |                                   | in TD3 paper.
    9  | ``learn.noise_``    dict        | dict(min=-0.5,    | Limit for range of target         |
       | ``range``                       |      max=0.5,)    | policy smoothing noise,           |
       |                                 |                   | aka. noise_clip.                  |
    10 | ``learn.-``         bool        False               | Determine whether to ignore       | Use ignore_done only
       | ``ignore_done``                                     | done flag.                        | in halfcheetah env.
    11 | ``learn.-``         float       0.005               | Used for soft update of the       | aka. Interpolation
       | ``target_theta``                                    | target network.                   | factor in polyak aver
       |                                                     |                                   | aging for target
       |                                                     |                                   | networks.
    12 | ``collect.-``       float       0.1                 | Used for add noise during co-     | Sample noise from dis
       | ``noise_sigma``                                     | llection, through controlling     | tribution, Ornstein-
       |                                                     | the sigma of distribution         | Uhlenbeck process in
       |                                                     |                                   | DDPG paper, Guassian
       |                                                     |                                   | process in ours.
    == ====================  ========    ==================  =================================   =======================
   """

    # You can refer to DDPG's default config for more details.
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='td3',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in TD3.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default False in TD3.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        # (str) Action space type
        action_space='continuous',  # ['continuous', 'hybrid']
        # (bool) Whether use batch normalization for reward
        reward_batch_norm=False,
        original_action_shape=2,
        model=dict(
            # (bool) Whether to use two critic networks or only one.
            # Clipped Double Q-Learning for Actor-Critic in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            twin_critic=True,
        ),
        learn=dict(
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # (float) Learning rates for critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=2,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            noise=True,
            # (float) Sigma for smoothing noise added to target policy.
            noise_sigma=0.2,
            # (dict) Limit for range of target policy smoothing noise, aka. noise_clip.
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(
            evaluator=dict(
                # (int) Evaluate every "eval_freq" training iterations.
                eval_freq=5000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer.
                replay_buffer_size=100000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config, main and target models.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
        )
        self._reward_batch_norm = self._cfg.reward_batch_norm

        self._gamma = self._cfg.learn.discount_factor
        self._actor_update_freq = self._cfg.learn.actor_update_freq
        self._twin_critic = self._cfg.model.twin_critic  # True for TD3, False for DDPG

        # main and target models
        self._target_model = copy.deepcopy(self._model)
        if self._cfg.action_space == 'hybrid':
            self._target_model = model_wrap(self._target_model, wrapper_name='hybrid_argmax_sample')
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        if self._cfg.learn.noise:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.learn.noise_sigma
                },
                noise_range=self._cfg.learn.noise_range
            )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        if self._cfg.action_space == 'hybrid':
            self._learn_model = model_wrap(self._learn_model, wrapper_name='hybrid_argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0  # count iterations
        # action_shape, obs_shape, latent_action_dim, hidden_size_list
        self._vae_model = VanillaVAE(
            self._cfg.original_action_shape, self._cfg.model.obs_shape, self._cfg.model.action_shape, [256, 256]
        )
        # self._vae_model = VanillaVAE(2, 8, 6, [256, 256])

        self._optimizer_vae = Adam(
            self._vae_model.parameters(),
            lr=self._cfg.learn.learning_rate_vae,
        )
        self._running_mean_std_predict_loss = RunningMeanStd(epsilon=1e-4)
        self.c_percentage_bound_lower = -1 * torch.ones([6])
        self.c_percentage_bound_upper = torch.ones([6])

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        # warmup phase
        if 'warm_up' in data[0].keys() and data[0]['warm_up'] is True:
            loss_dict = {}
            data = default_preprocess_learn(
                data,
                use_priority=self._cfg.priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=False
            )
            if self._cuda:
                data = to_device(data, self._device)

            # ====================
            # train vae
            # ====================
            result = self._vae_model({'action': data['action'], 'obs': data['obs']})

            result['original_action'] = data['action']
            result['true_residual'] = data['next_obs'] - data['obs']

            vae_loss = self._vae_model.loss_function(result, kld_weight=0.01, predict_weight=0.01)  # TODO(pu): weight

            loss_dict['vae_loss'] = vae_loss['loss'].item()
            loss_dict['reconstruction_loss'] = vae_loss['reconstruction_loss'].item()
            loss_dict['kld_loss'] = vae_loss['kld_loss'].item()
            loss_dict['predict_loss'] = vae_loss['predict_loss'].item()
            self._running_mean_std_predict_loss.update(vae_loss['predict_loss'].unsqueeze(-1).cpu().detach().numpy())

            # vae update
            self._optimizer_vae.zero_grad()
            vae_loss['loss'].backward()
            self._optimizer_vae.step()
            # For compatibility
            loss_dict['actor_loss'] = torch.Tensor([0]).item()
            loss_dict['critic_loss'] = torch.Tensor([0]).item()
            loss_dict['critic_twin_loss'] = torch.Tensor([0]).item()
            loss_dict['total_loss'] = torch.Tensor([0]).item()
            q_value_dict = {}
            q_value_dict['q_value'] = torch.Tensor([0]).item()
            q_value_dict['q_value_twin'] = torch.Tensor([0]).item()
            return {
                'cur_lr_actor': self._optimizer_actor.defaults['lr'],
                'cur_lr_critic': self._optimizer_critic.defaults['lr'],
                'action': torch.Tensor([0]).item(),
                'priority': torch.Tensor([0]).item(),
                'td_error': torch.Tensor([0]).item(),
                **loss_dict,
                **q_value_dict,
            }
        else:
            self._forward_learn_cnt += 1
            loss_dict = {}
            q_value_dict = {}
            data = default_preprocess_learn(
                data,
                use_priority=self._cfg.priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=False
            )
            if data['vae_phase'][0].item() is True:
                if self._cuda:
                    data = to_device(data, self._device)

                # ====================
                # train vae
                # ====================
                result = self._vae_model({'action': data['action'], 'obs': data['obs']})

                result['original_action'] = data['action']
                result['true_residual'] = data['next_obs'] - data['obs']

                # latent space constraint (LSC)
                # NOTE: using tanh is important, update latent_action using z, shape (128,6)
                data['latent_action'] = torch.tanh(result['z'].clone().detach())  # NOTE: tanh
                # data['latent_action'] = result['z'].clone().detach()
                self.c_percentage_bound_lower = data['latent_action'].sort(dim=0)[0][int(
                    result['recons_action'].shape[0] * 0.02
                ), :]  # values, indices
                self.c_percentage_bound_upper = data['latent_action'].sort(
                    dim=0
                )[0][int(result['recons_action'].shape[0] * 0.98), :]

                vae_loss = self._vae_model.loss_function(
                    result, kld_weight=0.01, predict_weight=0.01
                )  # TODO(pu): weight

                loss_dict['vae_loss'] = vae_loss['loss']
                loss_dict['reconstruction_loss'] = vae_loss['reconstruction_loss']
                loss_dict['kld_loss'] = vae_loss['kld_loss']
                loss_dict['predict_loss'] = vae_loss['predict_loss']

                # vae update
                self._optimizer_vae.zero_grad()
                vae_loss['loss'].backward()
                self._optimizer_vae.step()

                return {
                    'cur_lr_actor': self._optimizer_actor.defaults['lr'],
                    'cur_lr_critic': self._optimizer_critic.defaults['lr'],
                    # 'q_value': np.array(q_value).mean(),
                    'action': torch.Tensor([0]).item(),
                    'priority': torch.Tensor([0]).item(),
                    'td_error': torch.Tensor([0]).item(),
                    **loss_dict,
                    **q_value_dict,
                }

            else:
                # ====================
                # critic learn forward
                # ====================
                self._learn_model.train()
                self._target_model.train()
                next_obs = data['next_obs']
                reward = data['reward']

                # ====================
                # relabel latent action
                # ====================
                if self._cuda:
                    data = to_device(data, self._device)
                result = self._vae_model({'action': data['action'], 'obs': data['obs']})
                true_residual = data['next_obs'] - data['obs']

                # Representation shift correction (RSC)
                for i in range(result['recons_action'].shape[0]):
                    if F.mse_loss(result['prediction_residual'][i],
                                  true_residual[i]).item() > 4 * self._running_mean_std_predict_loss.mean:
                        # NOTE: using tanh is important, update latent_action using z
                        data['latent_action'][i] = torch.tanh(result['z'][i].clone().detach())  # NOTE: tanh
                        # data['latent_action'][i] = result['z'][i].clone().detach()

                # update all latent action
                # data['latent_action'] = torch.tanh(result['z'].clone().detach())

                if self._reward_batch_norm:
                    reward = (reward - reward.mean()) / (reward.std() + 1e-8)

                # current q value
                q_value = self._learn_model.forward(
                    {
                        'obs': data['obs'],
                        'action': data['latent_action']
                    }, mode='compute_critic'
                )['q_value']
                q_value_dict = {}
                if self._twin_critic:
                    q_value_dict['q_value'] = q_value[0].mean()
                    q_value_dict['q_value_twin'] = q_value[1].mean()
                else:
                    q_value_dict['q_value'] = q_value.mean()
                # target q value.
                with torch.no_grad():
                    # NOTE: here  next_actor_data['action'] is latent action
                    next_actor_data = self._target_model.forward(next_obs, mode='compute_actor')
                    next_actor_data['obs'] = next_obs
                    target_q_value = self._target_model.forward(next_actor_data, mode='compute_critic')['q_value']
                if self._twin_critic:
                    # TD3: two critic networks
                    target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
                    # critic network1
                    td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
                    critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
                    loss_dict['critic_loss'] = critic_loss
                    # critic network2(twin network)
                    td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
                    critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
                    loss_dict['critic_twin_loss'] = critic_twin_loss
                    td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
                else:
                    # DDPG: single critic network
                    td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
                    critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
                    loss_dict['critic_loss'] = critic_loss
                # ================
                # critic update
                # ================
                self._optimizer_critic.zero_grad()
                for k in loss_dict:
                    if 'critic' in k:
                        loss_dict[k].backward()
                self._optimizer_critic.step()
                # ===============================
                # actor learn forward and update
                # ===============================
                # actor updates every ``self._actor_update_freq`` iters
                if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
                    # NOTE: actor_data['action] is latent action
                    actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
                    actor_data['obs'] = data['obs']
                    if self._twin_critic:
                        actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0].mean()
                    else:
                        actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()

                    loss_dict['actor_loss'] = actor_loss
                    # actor update
                    self._optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self._optimizer_actor.step()
                # =============
                # after update
                # =============
                loss_dict['total_loss'] = sum(loss_dict.values())
                # self._forward_learn_cnt += 1
                self._target_model.update(self._learn_model.state_dict())
                if self._cfg.action_space == 'hybrid':
                    action_log_value = -1.  # TODO(nyz) better way to viz hybrid action
                else:
                    action_log_value = data['action'].mean()

                return {
                    'cur_lr_actor': self._optimizer_actor.defaults['lr'],
                    'cur_lr_critic': self._optimizer_critic.defaults['lr'],
                    'action': action_log_value,
                    'priority': td_error_per_sample.abs().tolist(),
                    'td_error': td_error_per_sample.abs().mean(),
                    **loss_dict,
                    **q_value_dict,
                }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_actor': self._optimizer_actor.state_dict(),
            'optimizer_critic': self._optimizer_critic.state_dict(),
            'vae_model': self._vae_model.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self._optimizer_critic.load_state_dict(state_dict['optimizer_critic'])
        self._vae_model.load_state_dict(state_dict['vae_model'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        # collect model
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma
            },
            noise_range=None
        )
        if self._cfg.action_space == 'hybrid':
            self._collect_model = model_wrap(self._collect_model, wrapper_name='hybrid_eps_greedy_multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, **kwargs) -> dict:
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
            output = self._collect_model.forward(data, mode='compute_actor', **kwargs)
            output['latent_action'] = output['action']

            # latent space constraint (LSC)
            for i in range(output['action'].shape[-1]):
                output['action'][:, i].clamp_(
                    self.c_percentage_bound_lower[i].item(), self.c_percentage_bound_upper[i].item()
                )

            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            output['action'] = self._vae_model.decode_with_obs(output['action'], data)['reconstruction_action']

        # NOTE: add noise in the original actions
        from ding.rl_utils.exploration import GaussianNoise
        action = output['action']
        gaussian_noise = GaussianNoise(mu=0.0, sigma=0.1)
        noise = gaussian_noise(output['action'].shape, output['action'].device)
        if self._cfg.learn.noise_range is not None:
            noise = noise.clamp(self._cfg.learn.noise_range['min'], self._cfg.learn.noise_range['max'])
        action += noise
        self.action_range = {'min': -1, 'max': 1}
        if self.action_range is not None:
            action = action.clamp(self.action_range['min'], self.action_range['max'])
        output['action'] = action

        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        if 'latent_action' in model_output.keys():
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': model_output['action'],
                'latent_action': model_output['latent_action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        else:  # if random collect at fist
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': model_output['action'],
                'latent_action': 999,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        if self._cfg.action_space == 'hybrid':
            transition['logit'] = model_output['logit']
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
        if self._cfg.action_space == 'hybrid':
            self._eval_model = model_wrap(self._eval_model, wrapper_name='hybrid_argmax_sample')
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
            output = self._eval_model.forward(data, mode='compute_actor')
            output['latent_action'] = output['action']

            # latent space constraint (LSC)
            for i in range(output['action'].shape[-1]):
                output['action'][:, i].clamp_(
                    self.c_percentage_bound_lower[i].item(), self.c_percentage_bound_upper[i].item()
                )

            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            output['action'] = self._vae_model.decode_with_obs(output['action'], data)['reconstruction_action']
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac', ['ding.model.template.qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' names if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'q_value_twin',
            'action', 'td_error', 'vae_loss', 'reconstruction_loss', 'kld_loss', 'predict_loss'
        ]
        if self._twin_critic:
            ret += ['critic_twin_loss']
        return ret
