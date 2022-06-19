from collections import namedtuple, Union
from typing import List, Dict, Any, Tuple
from easydict import EasyDict

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.model import model_wrap
from ding.rl_utils import vtrace_data, vtrace_error, get_train_sample
from ding.torch_utils import Adam, RMSprop, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
from ding.policy.sac import SACPolicy

from ding.rl_utils.isw import compute_importance_weights
from ding.rl_utils.vtrace import vtrace_nstep_return


@POLICY_REGISTRY.register('impala')
class IMPALAPolicy(Policy):
    r"""
    Overview:
        Policy class of IMPALA algorithm.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      impala         | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority

        5  | ``priority_``      bool     False          | Whether use Importance Sampling Weight | If True, priority
           | ``IS_weight``                              |                                        | must be True
        6  ``unroll_len``       int      32             | trajectory length to calculate v-trace
                                                        | target
        7  | ``learn.update``   int      4              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        type='impala',
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        # here we follow ppo serial pipeline, the original is False
        on_policy=False,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous']
        action_space='discrete',
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) the trajectory length to calculate v-trace target
        unroll_len=32,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            # (int) the number of data for a train iteration
            batch_size=16,
            learning_rate=0.0005,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (float) clip ratio of importance weights
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling
            rho_pg_clip_ratio=1.0,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            # n_sample=16,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
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
        grad_clip_type = self._cfg.learn.get("grad_clip_type", None)
        clip_value = self._cfg.learn.get("clip_value", None)
        optim_type = self._cfg.learn.get("optim", "adam")
        if optim_type == 'rmsprop':
            self._optimizer = RMSprop(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        elif optim_type == 'adam':
            self._optimizer = Adam(
                self._model.parameters(),
                grad_clip_type=grad_clip_type,
                clip_value=clip_value,
                lr=self._cfg.learn.learning_rate
            )
        else:
            raise NotImplementedError
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        self._action_shape = self._cfg.model.action_shape
        self._unroll_len = self._cfg.unroll_len

        # Algorithm config
        self._action_space = self._cfg.action_space
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._gamma = self._cfg.learn.discount_factor
        self._lambda = self._cfg.learn.lambda_
        self._rho_clip_ratio = self._cfg.learn.rho_clip_ratio
        self._c_clip_ratio = self._cfg.learn.c_clip_ratio
        self._rho_pg_clip_ratio = self._cfg.learn.rho_pg_clip_ratio

        # Main model
        self._learn_model.reset()

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
        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)
        data['obs_plus_1'] = torch.cat((data['obs'] + data['next_obs'][-1:]), dim=0)  # shape (T+1)*B,env_obs_shape
        if self._action_space == 'discrete':
            data['logit'] = torch.cat(
                data['logit'], dim=0
            ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
            data['action'] = torch.cat(data['action'], dim=0).reshape(self._unroll_len, -1)  # shape T,B,
        else:
            data['logit'][0] = torch.cat(
                data['logit'][0], dim=0
            ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
            data['logit'][1] = torch.cat(
                data['logit'][1], dim=0
            ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
            data['action'] = torch.cat(
                data['action'], dim=0
            ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
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
            - necessary: ``cur_lr``, ``total_loss``, ``policy_loss`,``value_loss``,``entropy_loss``
            - optional: ``priority``
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # IMPALA forward
        # ====================
        self._learn_model.train()
        output = self._learn_model.forward(data['obs_plus_1'], mode='compute_actor_critic')
        target_logit, behaviour_logit, actions, values, rewards, weights = self._reshape_data(output, data)
        # Calculate vtrace error
        data = vtrace_data(target_logit, behaviour_logit, actions, values, rewards, weights)
        g, l, r, c, rg = self._gamma, self._lambda, self._rho_clip_ratio, self._c_clip_ratio, self._rho_pg_clip_ratio
        vtrace_loss = vtrace_error(data, g, l, r, c, rg, self._action_space)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = vtrace_loss.policy_loss + wv * vtrace_loss.value_loss - we * vtrace_loss.entropy_loss
        # ====================
        # IMPALA update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': vtrace_loss.policy_loss.item(),
            'value_loss': vtrace_loss.value_loss.item(),
            'entropy_loss': vtrace_loss.entropy_loss.item(),
        }

    def _reshape_data(self, output: Dict[str, Any], data: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
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
            - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - values (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        if self._action_space == 'discrete':
            target_logit = output['logit'].reshape(self._unroll_len + 1, -1,
                                                   self._action_shape)[:-1]  # shape (T+1),B,env_obs_shape
        else:
            target_logit = output['logit']
            target_logit[0] = target_logit[0].reshape(self._unroll_len + 1, -1,
                                                      self._action_shape)[:-1]  # shape (T+1),B,env_obs_shape
            target_logit[1] = target_logit[1].reshape(self._unroll_len + 1, -1,
                                                      self._action_shape)[:-1]  # shape (T+1),B,env_obs_shape
        behaviour_logit = data['logit']  # shape T,B
        actions = data['action']  # shape T,B
        values = output['value'].reshape(self._unroll_len + 1, -1)  # shape T+1,B,env_action_shape
        rewards = data['reward']  # shape T,B
        weights_ = 1 - data['done']  # shape T,B
        weights = torch.ones_like(rewards)  # shape T,B
        values[1:] = values[1:] * weights_
        weights[1:] = weights_[:-1]
        rewards = rewards * weights  # shape T,B
        return target_logit, behaviour_logit, actions, values, rewards, weights

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
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
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model.
            Use multinomial_sample to choose action.
        """
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
        return 'vac', ['ding.model.template.vac']

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
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']


@POLICY_REGISTRY.register('qimpala')
class QIMPALAPolicy(SACPolicy, IMPALAPolicy):

    config = dict(
        type='qimpala',
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous']
        action_space='continuous',
        # (int) the trajectory length to calculate v-trace target
        unroll_len=32,
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (float) clip ratio of importance weights
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            # n_sample=16,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=1000,
            max_use=16,
        ), ),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        pass

    def _init_learn(self) -> None:
        SACPolicy._init_learn(self)
        self._auto_alpha = False
        self._value_network = False

        self._action_space = self._cfg.action_space
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._gamma = self._cfg.learn.discount_factor
        self._lambda = self._cfg.learn.lambda_
        self._rho_clip_ratio = self._cfg.learn.rho_clip_ratio
        self._c_clip_ratio = self._cfg.learn.c_clip_ratio

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
            - necessary: ``cur_lr``, ``total_loss``, ``policy_loss`,``value_loss``,``entropy_loss``
            - optional: ``priority``
        """
        loss_dict = {}
        data['raw_action'] = data['action']
        data = self._data_preprocess_learn(data)
        # ====================
        # IMPALA forward
        # ====================
        self._learn_model.train()
        # data['obs_plus_1']: (T+1)*B,obs_shape
        # output['logit']['mu']: (T+1)*B,action_shape
        # output['logit']['sigma']: (T+1)*B,action_shape
        output = self._learn_model.forward(data['obs_plus_1'], mode='compute_actor')
        dist = Independent(Normal(*output['logit']), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        log_prob = dist.log_prob(
            pred
        ) + 2 * (pred + torch.nn.functional.softplus(-2. * pred) - torch.log(torch.tensor(2.))).sum(-1)

        eval_data = {'obs': data['obs_plus_1'], 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        loss_dict['policy_loss'] = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        # data['next_obs'][-1]: B,obs_shape
        with torch.no_grad():
            # B,action_shape
            last_action = action[-1].detach()
            # data['action_plus_1'] = (T+1)*B,action_shape
            data['action_plus_1'] = torch.cat((data['raw_action'] + [last_action]), dim=0)
            eval_data = {'obs': data['obs_plus_1'], 'action': data['action_plus_1']}
            # output['q_value']: (T+1)*B
            output.update(self._target_model.forward(eval_data, mode='compute_critic'))
            if self._twin_critic:
                output['value'] = torch.min(output['q_value'][0], output['q_value'][1]) - self._alpha * log_prob
            else:
                output['value'] = output['q_value'] - self.alpha * log_prob
            data['reward'] -= (self.alpha * log_prob).reshape(self._unroll_len + 1, -1)[:-1]
            target_logit, behaviour_logit, actions, values, rewards, weights = self._reshape_data(output, data)

            IS = compute_importance_weights(target_logit, behaviour_logit, actions, self._action_space)
            rhos = torch.clamp(IS, max=self._rho_clip_ratio)
            cs = torch.clamp(IS, max=self._c_clip_ratio)
            # (T,B)
            return_ = vtrace_nstep_return(rhos, cs, rewards, values, self._gamma, self._lambda)

        eval_data = {'obs': data['obs_plus_1'], 'action': data['action_plus_1']}
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        q_value = q_value.reshape(self._unroll_len + 1, -1)
        loss_dict['critic_loss'] = (F.mse_loss(q_value[:-1], return_, reduction='none') * weights).mean()

        # ====================
        # IMPALA update
        # ====================
        # todo: alpha loss
        # update critic
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_q.step()
        # update policy
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()
        # update temperature
        # self._alpha_optim.zero_grad()
        # loss_dict['alpha_loss'].backward()
        # self._alpha_optim.step()
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'alpha': self._alpha.item(),
            'impala_return': self._alpha.item(),
            **loss_dict,
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        SACPolicy._state_dict_learn(self)

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        SACPolicy._load_state_dict_learn(self, state_dict)

    def _init_collect(self) -> None:
        SACPolicy._init_collect(self)

    def _forward_collect(self, data: dict) -> dict:
        SACPolicy._forward_collect(self)

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        # DynaWorldModel.__init__(self, cfg, env, tb_logger)
        IMPALAPolicy._process_transition(self, obs, policy_output, timestep)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        IMPALAPolicy._get_train_sample(self, data)

    def _init_eval(self) -> None:
        SACPolicy._init_eval(self)

    def _forward_eval(self, data: dict) -> dict:
        SACPolicy._forward_eval(self, data)

    def default_model(self) -> Tuple[str, List[str]]:
        SACPolicy.default_model(self)

    def _monitor_vars_learn(self) -> List[str]:
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        return [
            'policy_loss',
            'critic_loss',
            'cur_lr_q',
            'cur_lr_p',
            'alpha',
            'impala_return',
        ] + alpha_loss
