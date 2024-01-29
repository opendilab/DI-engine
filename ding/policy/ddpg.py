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


@POLICY_REGISTRY.register('ddpg')
class DDPGPolicy(Policy):
    """
    Overview:
        Policy class of DDPG algorithm. Paper link: https://arxiv.org/abs/1509.02971.

    Config:
        == ====================  ========    =============  =================================   =======================
        ID Symbol                Type        Default Value  Description                         Other(Shape)
        == ====================  ========    =============  =================================   =======================
        1  | ``type``            str         ddpg           | RL policy register name, refer    | this arg is optional,
           |                                                | to registry ``POLICY_REGISTRY``   | a placeholder
        2  | ``cuda``            bool        False          | Whether to use cuda for network   |
        3  | ``random_``         int         25000          | Number of randomly collected      | Default to 25000 for
           | ``collect_size``                               | training samples in replay        | DDPG/TD3, 10000 for
           |                                                | buffer when training starts.      | sac.
        4  | ``model.twin_``     bool        False          | Whether to use two critic         | Default False for
           | ``critic``                                     | networks or only one.             | DDPG, Clipped Double
           |                                                |                                   | Q-learning method in
           |                                                |                                   | TD3 paper.
        5  | ``learn.learning``  float       1e-3           | Learning rate for actor           |
           | ``_rate_actor``                                | network(aka. policy).             |
        6  | ``learn.learning``  float       1e-3           | Learning rates for critic         |
           | ``_rate_critic``                               | network (aka. Q-network).         |
        7  | ``learn.actor_``    int         2              | When critic network updates       | Default 1 for DDPG,
           | ``update_freq``                                | once, how many times will actor   | 2 for TD3. Delayed
           |                                                | network update.                   | Policy Updates method
           |                                                |                                   | in TD3 paper.
        8  | ``learn.noise``     bool        False          | Whether to add noise on target    | Default False for
           |                                                | network's action.                 | DDPG, True for TD3.
           |                                                |                                   | Target Policy Smoo-
           |                                                |                                   | thing Regularization
           |                                                |                                   | in TD3 paper.
        9  | ``learn.-``         bool        False          | Determine whether to ignore       | Use ignore_done only
           | ``ignore_done``                                | done flag.                        | in halfcheetah env.
        10 | ``learn.-``         float       0.005          | Used for soft update of the       | aka. Interpolation
           | ``target_theta``                               | target network.                   | factor in polyak aver-
           |                                                |                                   | aging for target
           |                                                |                                   | networks.
        11 | ``collect.-``       float       0.1            | Used for add noise during co-     | Sample noise from dis-
           | ``noise_sigma``                                | llection, through controlling     | tribution, Ornstein-
           |                                                | the sigma of distribution         | Uhlenbeck process in
           |                                                |                                   | DDPG paper, Gaussian
           |                                                |                                   | process in ours.
        == ====================  ========    =============  =================================   =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ddpg',
        # (bool) Whether to use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy). Default False in DDPG.
        on_policy=False,
        # (bool) Whether to enable priority experience sample.
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        # (bool) Whether to need policy data in process transition.
        transition_with_policy_data=False,
        # (str) Action space type, including ['continuous', 'hybrid'].
        action_space='continuous',
        # (bool) Whether use batch normalization for reward.
        reward_batch_norm=False,
        # (bool) Whether to enable multi-agent training setting.
        multi_agent=False,
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
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
            # (float) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=1,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            noise=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=1,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(),  # for compability
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=100000,
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
        if self._cfg.multi_agent:
            return 'continuous_maqac', ['ding.model.template.maqac']
        else:
            return 'continuous_qac', ['ding.model.template.qac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For DDPG, it mainly \
            contains two optimizers, algorithm-specific arguments such as gamma and twin_critic, main and target model.
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
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        if self._cfg.action_space == 'hybrid':
            self._learn_model = model_wrap(self._learn_model, wrapper_name='hybrid_argmax_sample')
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
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                For DDPG, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
                and ``logit`` which is used for hybrid action space.
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
            For more detailed examples, please refer to our unittest for DDPGPolicy: ``ding.policy.tests.test_ddpg``.
        """
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
        # critic learn forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        next_obs = data['next_obs']
        reward = data['reward']
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # target q value.
        with torch.no_grad():
            next_actor_data = self._target_model.forward(next_obs, mode='compute_actor')
            next_actor_data['obs'] = next_obs
            target_q_value = self._target_model.forward(next_actor_data, mode='compute_critic')['q_value']

        q_value_dict = {}
        target_q_value_dict = {}

        if self._twin_critic:
            # TD3: two critic networks
            target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
            q_value_dict['q_value'] = q_value[0].mean().data.item()
            q_value_dict['q_value_twin'] = q_value[1].mean().data.item()
            target_q_value_dict['target q_value'] = target_q_value.mean().data.item()
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
            q_value_dict['q_value'] = q_value.mean().data.item()
            target_q_value_dict['target q_value'] = target_q_value.mean().data.item()
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
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        if self._cfg.action_space == 'hybrid':
            action_log_value = -1.  # TODO(nyz) better way to viz hybrid action
        else:
            action_log_value = data['action'].mean()
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            # 'q_value': np.array(q_value).mean(),
            'action': action_log_value,
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.abs().mean(),
            **loss_dict,
            **q_value_dict,
            **target_q_value_dict,
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizers.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_actor': self._optimizer_actor.state_dict(),
            'optimizer_critic': self._optimizer_critic.state_dict(),
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
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self._optimizer_critic.load_state_dict(state_dict['optimizer_critic'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For DDPG, it contains the \
            collect_model to balance the exploration and exploitation with the perturbed noise mechanism, and other \
            algorithm-specific arguments such as unroll_len. \
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
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
                dict is the same as the input data, i.e., environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DDPGPolicy: ``ding.policy.tests.test_ddpg``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor', **kwargs)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For DDPG, it contains obs, next_obs, action, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For DDPG, it contains the action and the logit of the action (in hybrid action space).
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        if self._cfg.action_space == 'hybrid':
            transition['logit'] = policy_output['logit']
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In DDPG, a train sample is a processed transition (unroll_len=1).
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
            Initialize the eval mode of policy, including related attributes and modules. For DDPG, it contains the \
            eval model to greedily select action type with argmax q_value mechanism for hybrid action space. \
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        if self._cfg.action_space == 'hybrid':
            self._eval_model = model_wrap(self._eval_model, wrapper_name='hybrid_argmax_sample')
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
            For more detailed examples, please refer to our unittest for DDPGPolicy: ``ding.policy.tests.test_ddpg``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
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
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'q_value_twin',
            'action', 'td_error'
        ]
        if self._twin_critic:
            ret += ['critic_twin_loss']
        return ret
