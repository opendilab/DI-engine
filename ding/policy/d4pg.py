from typing import List, Dict, Any, Tuple, Union
import torch
import copy

from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .ddpg import DDPGPolicy
from .common_utils import default_preprocess_learn
import numpy as np


@POLICY_REGISTRY.register('d4pg')
class D4PGPolicy(DDPGPolicy):
    """
    Overview:
        Policy class of D4PG algorithm. D4PG is a variant of DDPG, which uses distributional critic. \
        The distributional critic is implemented by using quantile regression. \
        Paper link: https://arxiv.org/abs/1804.08617.

    Property:
        learn_mode, collect_mode, eval_mode
    Config:
        == ====================  ========    =============  =================================   =======================
        ID Symbol                Type        Default Value  Description                         Other(Shape)
        == ====================  ========    =============  =================================   =======================
        1  ``type``              str         d4pg           | RL policy register name, refer    | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY``   | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network   |
        3  | ``random_``         int         25000          | Number of randomly collected      | Default to 25000 for
           | ``collect_size``                               | training samples in replay        | DDPG/TD3, 10000 for
           |                                                | buffer when training starts.      | sac.
        5  | ``learn.learning``  float       1e-3           | Learning rate for actor           |
           | ``_rate_actor``                                | network(aka. policy).             |
        6  | ``learn.learning``  float       1e-3           | Learning rates for critic         |
           | ``_rate_critic``                               | network (aka. Q-network).         |
        7  | ``learn.actor_``    int         1              | When critic network updates       | Default 1
           | ``update_freq``                                | once, how many times will actor   |
           |                                                | network update.                   |
        8  | ``learn.noise``     bool        False          | Whether to add noise on target    | Default False for
           |                                                | network's action.                 | D4PG.
           |                                                |                                   | Target Policy Smoo-
           |                                                |                                   | thing Regularization
           |                                                |                                   | in TD3 paper.
        9  | ``learn.-``         bool        False          | Determine whether to ignore       | Use ignore_done only
           | ``ignore_done``                                | done flag.                        | in halfcheetah env.
        10 | ``learn.-``         float       0.005          | Used for soft update of the       | aka. Interpolation
           | ``target_theta``                               | target network.                   | factor in polyak aver
           |                                                |                                   | aging for target
           |                                                |                                   | networks.
        11 | ``collect.-``       float       0.1            | Used for add noise during co-     | Sample noise from dis
           | ``noise_sigma``                                | llection, through controlling     | tribution, Gaussian
           |                                                | the sigma of distribution         | process.
        12 | ``model.v_min``      float      -10            | Value of the smallest atom        |
           |                                                | in the support set.               |
        13 | ``model.v_max``      float      10             | Value of the largest atom         |
           |                                                | in the support set.               |
        14 | ``model.n_atom``     int        51             | Number of atoms in the support    |
           |                                                | set of the value distribution.    |
        15 | ``nstep``            int        3, [1, 5]      | N-step reward discount sum for    |
           |                                                | target q_value estimation         |
        16 | ``priority``         bool       True           | Whether use priority(PER)         | priority sample,
                                                                                                | update priority
        == ====================  ========    =============  =================================   =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='d4pg',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in D4PG.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default True in D4PG.
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in D4PG.
        random_collect_size=25000,
        # (int) N-step reward for target q_value estimation
        nstep=3,
        # (str) Action space type
        action_space='continuous',  # ['continuous', 'hybrid']
        # (bool) Whether use batch normalization for reward
        reward_batch_norm=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=False,
        model=dict(
            # (float) Value of the smallest atom in the support set.
            # Default to -10.0.
            v_min=-10,
            # (float) Value of the smallest atom in the support set.
            # Default to 10.0.
            v_max=10,
            # (int) Number of atoms in the support set of the
            # value distribution. Default to 51.
            n_atom=51
        ),
        learn=dict(

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # Learning rates for critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
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
            actor_update_freq=1,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper.
            noise=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] should be set
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return the default neural network model class for D4PGPolicy. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'qac_dist', ['ding.model.template.qac_dist']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the D4PG policy's learning mode, which involves setting up key components \
            specific to the D4PG algorithm. This includes creating separate optimizers for the actor \
            and critic networks, a distinctive trait of D4PG's actor-critic approach, and configuring \
            algorithm-specific parameters such as v_min, v_max, and n_atom for the distributional aspect \
            of the critic. Additionally, the method sets up the target model with momentum-based updates, \
            crucial for stabilizing learning, and optionally integrates noise into the target model for \
            effective exploration. This method is invoked during the '__init__' if 'learn' is specified \
            in 'enable_field'.

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
        self._nstep = self._cfg.nstep
        self._actor_update_freq = self._cfg.learn.actor_update_freq

        # main and target models
        self._target_model = copy.deepcopy(self._model)
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
        self._learn_model.reset()
        self._target_model.reset()

        self._v_max = self._cfg.model.v_max
        self._v_min = self._cfg.model.v_min
        self._n_atom = self._cfg.model.n_atom

        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as different loss, actor and critic lr.
        Arguments:
            - data (:obj:`dict`): Input data used for policy forward, including the \
                collected training samples from replay buffer. For each element in dict, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For D4PG, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``reward``, ``next_obs``. Sometimes, it also contains other keys such as ``weight``.

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The output result dict of forward learn, containing at \
                least the "cur_lr_actor", "cur_lr_critic", "different losses", "q_value", "action", "priority", \
                keys. Additionally, loss_dict also contains other keys, which are mainly used for monitoring and \
                debugging. "q_value_dict" is used to record the q_value statistics.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for D4PGPolicy: ``ding.policy.tests.test_d4pg``.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # critic learn forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._learn_model.forward(data, mode='compute_critic')
        q_value_dict = {}
        q_dist = q_value['distribution']
        q_value_dict['q_value'] = q_value['q_value'].mean()
        # target q value.
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_dist = self._target_model.forward(next_data, mode='compute_critic')['distribution']

        value_gamma = data.get('value_gamma')
        action_index = np.zeros(next_action.shape[0])
        # since the action is a scalar value, action index is set to 0 which is the only possible choice
        td_data = dist_nstep_td_data(
            q_dist, target_q_dist, action_index, action_index, reward, data['done'], data['weight']
        )
        critic_loss, td_error_per_sample = dist_nstep_td_error(
            td_data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep, value_gamma=value_gamma
        )
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
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            'q_value': q_value['q_value'].mean().item(),
            'action': data['action'].mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict,
            **q_value_dict,
        }

    def _get_train_sample(self, traj: list) -> Union[None, List[Any]]:
        """
        Overview:
            Process the data of a given trajectory (transitions, a list of transition) into a list of sample that \
            can be used for training directly. The sample is generated by the following steps: \
            1. Calculate the nstep return data. \
            2. Sample the data from the nstep return data. \
            3. Stack the data in the batch dimension. \
            4. Return the sample data. \
            For D4PG, the nstep return data is generated by ``get_nstep_return_data`` and the sample data is \
            generated by ``get_train_sample``.

        Arguments:
            - traj (:obj:`list`): The trajectory data (a list of transition), each element is \
            the same format as the return value of ``self._process_transition`` method.

        Returns:
            - samples (:obj:`dict`): The training samples generated, including at least the following keys: \
            ``'obs'``, ``'next_obs'``, ``'action'``, ``'reward'``, ``'done'``, ``'weight'``, ``'value_gamma'``. \
            For more information, please refer to the ``get_train_sample`` method.
        """
        data = get_nstep_return_data(traj, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        ret = ['cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'action']
        return ret
