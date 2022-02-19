from collections import namedtuple
from typing import List, Dict, Any, Tuple
import copy

import torch

from ding.model import model_wrap
from ding.rl_utils import get_train_sample, compute_q_retraces, acer_policy_error,\
     acer_value_error, acer_trust_region_update
from ding.torch_utils import Adam, RMSprop, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy

EPS = 1e-8


@POLICY_REGISTRY.register('acer')
class ACERPolicy(Policy):
    r"""
    Overview:
        Policy class of ACER algorithm.

    Config:
        == ======================= ======== ============== ===================================== =======================
        ID Symbol                  Type     Default Value  Description                           Other(Shape)
        == ======================= ======== ============== ===================================== =======================
        1  ``type``                str      acer           | RL policy register name, refer to   | this arg is optional,
                                                           | registry ``POLICY_REGISTRY``        | a placeholder
        2  ``cuda``                bool     False          | Whether to use cuda for network     | this arg can be diff-
                                                           |                                     | erent from modes
        3  ``on_policy``           bool     False          | Whether the RL algorithm is
                                                           | on-policy or off-policy
        4  ``trust_region``        bool     True           | Whether the RL algorithm use trust  |
                                                           | region constraint                   |
        5  ``trust_region_value``  float    1.0            | maximum range of the trust region   |
        6  ``unroll_len``          int      32             | trajectory length to calculate
                                                           | Q retrace target
        7   ``learn.update``       int      4              | How many updates(iterations) to     | this args can be vary
            ``per_collect``                                | train after collector's one         | from envs. Bigger val
                                                           |  collection. Only                   |
                                                           | valid in serial training            | means more off-policy
        8   ``c_clip_ratio``       float    1.0            | clip ratio of importance weights    |
        == ======================= ======== ============== ===================================== =======================
    """
    unroll_len = 32
    config = dict(
        type='acer',
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        # here we follow ppo serial pipeline, the original is False
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (str) the type of gradient clip method
            grad_clip_type=None,
            # (float) max value when ACER use gradient clip
            clip_value=None,
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            # (int) the number of data for a train iteration
            batch_size=16,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (int) the trajectory length to calculate v-trace target
            unroll_len=unroll_len,
            # (float) clip ratio of importance weights
            c_clip_ratio=10,
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            target_theta=0.01
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            # n_sample=16,
            # (int) the trajectory length to calculate v-trace target
            unroll_len=unroll_len,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            gae_lambda=0.95,
            collector=dict(
                type='sample',
                collect_print_freq=1000,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=1000,
            max_use=16,
        ), ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Initialize the optimizer, algorithm config and main model.
        """
        # Optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.clip_value
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
        )
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        self._action_shape = self._cfg.model.action_shape
        self._unroll_len = self._cfg.learn.unroll_len

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._gamma = self._cfg.learn.discount_factor
        # self._rho_clip_ratio = self._cfg.learn.rho_clip_ratio
        self._c_clip_ratio = self._cfg.learn.c_clip_ratio
        # self._rho_pg_clip_ratio = self._cfg.learn.rho_pg_clip_ratio
        self._use_trust_region = self._cfg.learn.trust_region
        self._trust_region_value = self._cfg.learn.trust_region_value
        # Main model
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]):
        """
        Overview:
            Data preprocess function of learn mode.
            Convert list trajectory data to to trajectory data, which is a dict of tensors.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): List type data, a list of data for training. Each list element is a \
            dict, whose values are torch.Tensor or np.ndarray or dict/list combinations, keys include at least 'obs',\
             'next_obs', 'logit', 'action', 'reward', 'done'
        Returns:
            - data (:obj:`dict`): Dict type data. Values are torch.Tensor or np.ndarray or dict/list combinations. \
        ReturnsKeys:
            - necessary: 'logit', 'action', 'reward', 'done', 'weight', 'obs_plus_1'.
            - optional and not used in later computation: 'obs', 'next_obs'.'IS', 'collect_iter', 'replay_unique_id', \
                'replay_buffer_idx', 'priority', 'staleness', 'use'.
        ReturnsShapes:
            - obs_plus_1 (:obj:`torch.FloatTensor`): :math:`(T * B, obs_shape)`, where T is timestep, B is batch size \
                and obs_shape is the shape of single env observation
            - logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim
            - action (:obj:`torch.LongTensor`): :math:`(T, B)`
            - reward (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - done (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weight (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        # shape (T+1)*B,env_obs_shape
        data['obs_plus_1'] = torch.cat((data['obs'] + data['next_obs'][-1:]), dim=0)
        data['logit'] = torch.cat(
            data['logit'], dim=0
        ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
        data['action'] = torch.cat(data['action'], dim=0).reshape(self._unroll_len, -1)  # shape T,B,
        data['done'] = torch.cat(data['done'], dim=0).reshape(self._unroll_len, -1).float()  # shape T,B,
        data['reward'] = torch.cat(data['reward'], dim=0).reshape(self._unroll_len, -1)  # shape T,B,
        data['weight'] = torch.cat(
            data['weight'], dim=0
        ).reshape(self._unroll_len, -1) if data['weight'] else None  # shape T,B
        return data

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        r"""
        Overview:
            Forward computation graph of learn mode(updating policy).
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): List type data, a list of data for training. Each list element is a \
            dict, whose values are torch.Tensor or np.ndarray or dict/list combinations, keys include at least 'obs',\
             'next_obs', 'logit', 'action', 'reward', 'done'
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: 'collect_iter', 'replay_unique_id', 'replay_buffer_idx', 'priority', 'staleness', 'use', 'IS'
        ReturnsKeys:
            - necessary: ``cur_lr_actor``, ``cur_lr_critic``, ``actor_loss`,``bc_loss``,``policy_loss``,\
                ``critic_loss``,``entropy_loss``
        """
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        action_data = self._learn_model.forward(data['obs_plus_1'], mode='compute_actor')
        q_value_data = self._learn_model.forward(data['obs_plus_1'], mode='compute_critic')
        avg_action_data = self._target_model.forward(data['obs_plus_1'], mode='compute_actor')

        target_logit, behaviour_logit, avg_logit, actions, q_values, rewards, weights = self._reshape_data(
            action_data, avg_action_data, q_value_data, data
        )
        # shape (T+1),B,env_action_shape
        target_logit = torch.log_softmax(target_logit, dim=-1)
        # shape T,B,env_action_shape
        behaviour_logit = torch.log_softmax(behaviour_logit, dim=-1)
        # shape (T+1),B,env_action_shape
        avg_logit = torch.log_softmax(avg_logit, dim=-1)
        with torch.no_grad():
            # shape T,B,env_action_shape
            ratio = torch.exp(target_logit[0:-1] - behaviour_logit)
            # shape (T+1),B,1
            v_pred = (q_values * torch.exp(target_logit)).sum(-1).unsqueeze(-1)
            # Calculate retrace
            q_retraces = compute_q_retraces(q_values, v_pred, rewards, actions, weights, ratio, self._gamma)

        # the terminal states' weights are 0. it needs to be shift to count valid state
        weights_ext = torch.ones_like(weights)
        weights_ext[1:] = weights[0:-1]
        weights = weights_ext
        q_retraces = q_retraces[0:-1]  # shape T,B,1
        q_values = q_values[0:-1]  # shape T,B,env_action_shape
        v_pred = v_pred[0:-1]  # shape T,B,1
        target_logit = target_logit[0:-1]  # shape T,B,env_action_shape
        avg_logit = avg_logit[0:-1]  # shape T,B,env_action_shape
        total_valid = weights.sum()  # 1
        # ====================
        # policy update
        # ====================
        actor_loss, bc_loss = acer_policy_error(
            q_values, q_retraces, v_pred, target_logit, actions, ratio, self._c_clip_ratio
        )
        actor_loss = actor_loss * weights.unsqueeze(-1)
        bc_loss = bc_loss * weights.unsqueeze(-1)
        dist_new = torch.distributions.categorical.Categorical(logits=target_logit)
        entropy_loss = (dist_new.entropy() * weights).unsqueeze(-1)  # shape T,B,1
        total_actor_loss = (actor_loss + bc_loss + self._entropy_weight * entropy_loss).sum() / total_valid
        self._optimizer_actor.zero_grad()
        actor_gradients = torch.autograd.grad(-total_actor_loss, target_logit, retain_graph=True)
        if self._use_trust_region:
            actor_gradients = acer_trust_region_update(
                actor_gradients, target_logit, avg_logit, self._trust_region_value
            )
        target_logit.backward(actor_gradients)
        self._optimizer_actor.step()

        # ====================
        # critic update
        # ====================
        critic_loss = (acer_value_error(q_values, q_retraces, actions) * weights.unsqueeze(-1)).sum() / total_valid
        self._optimizer_critic.zero_grad()
        critic_loss.backward()
        self._optimizer_critic.step()
        self._target_model.update(self._learn_model.state_dict())

        with torch.no_grad():
            kl_div = torch.exp(avg_logit) * (avg_logit - target_logit)
            kl_div = (kl_div.sum(-1) * weights).sum() / total_valid

        return {
            'cur_actor_lr': self._optimizer_actor.defaults['lr'],
            'cur_critic_lr': self._optimizer_critic.defaults['lr'],
            'actor_loss': (actor_loss.sum() / total_valid).item(),
            'bc_loss': (bc_loss.sum() / total_valid).item(),
            'policy_loss': total_actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': (entropy_loss.sum() / total_valid).item(),
            'kl_div': kl_div.item()
        }

    def _reshape_data(
            self, action_data: Dict[str, Any], avg_action_data: Dict[str, Any], q_value_data: Dict[str, Any],
            data: Dict[str, Any]
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        r"""
        Overview:
            Obtain weights for loss calculating, where should be 0 for done positions
            Update values and rewards with the weight
        Arguments:
            - output (:obj:`Dict[int, Any]`): Dict type data, output of learn_model forward. \
             Values are torch.Tensor or np.ndarray or dict/list combinations,keys are value, logit.
            - data (:obj:`Dict[int, Any]`): Dict type data, input of policy._forward_learn \
             Values are torch.Tensor or np.ndarray or dict/list combinations. Keys includes at \
             least ['logit', 'action', 'reward', 'done',]
        Returns:
            - data (:obj:`Tuple[Any]`): Tuple of target_logit, behaviour_logit, actions, \
             values, rewards, weights
        ReturnsShapes:
            - target_logit (:obj:`torch.FloatTensor`): :math:`((T+1), B, Obs_Shape)`, where T is timestep,\
             B is batch size and Obs_Shape is the shape of single env observation.
            - behaviour_logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim.
            - avg_action_logit (:obj:`torch.FloatTensor`): :math: `(T+1, B, N)`, where N is action dim.
            - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - values (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        target_logit = action_data['logit'].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        behaviour_logit = data['logit']  # shape T,B,env_action_shape
        avg_action_logit = avg_action_data['logit'].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        actions = data['action']  # shape T,B
        values = q_value_data['q_value'].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        rewards = data['reward']  # shape T,B
        weights_ = 1 - data['done']  # shape T,B
        weights = torch.ones_like(rewards)  # shape T,B
        weights = weights_
        return target_logit, behaviour_logit, avg_action_logit, actions, values, rewards, weights

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'actor_optimizer': self._optimizer_actor.state_dict(),
            'critic_optimizer': self._optimizer_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.
        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_actor.load_state_dict(state_dict['actor_optimizer'])
        self._optimizer_critic.load_state_dict(state_dict['critic_optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model.
            Use multinomial_sample to choose action.
        """
        self._collect_unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        r"""
        Overview:
            Forward computation graph of collect mode(collect training data).
        Arguments:
            - data (:obj:`Dict[int, Any]`): Dict type data, stacked env data for predicting \
            action, values are torch.Tensor or np.ndarray or dict/list combinations,keys \
            are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Dict[str,Any]]`): Dict of predicting policy_output(logit, action) for each env.
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly.
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): List of training samples.
        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procedure by overriding this two methods and collector \
            itself.
        """
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation,can be torch.Tensor or np.ndarray or dict/list combinations.
                - model_output (:obj:`dict`): Output of collect model, including ['logit','action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data, including at least ['obs','next_obs', 'logit',\
               'action','reward', 'done']
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': policy_output['logit'],
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model,
            and use argmax_sample to choose action.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        r"""
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
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
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        return 'acer', ['ding.model.template.acer']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names
        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For IMPALA, ``ding.model.interface.IMPALA``
        """
        return ['actor_loss', 'bc_loss', 'policy_loss', 'critic_loss', 'entropy_loss', 'kl_div']
