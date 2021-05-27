import math
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
import torch.nn as nn
import numpy as np

from nervex.torch_utils import Adam, to_device, one_hot
from nervex.model import model_wrap
from nervex.data import default_collate, default_decollate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
try:
    from app_zoo.gfootball.model.bots import FootballRuleBaseModel, FootballKaggle5thPlaceModel
except ImportError:
    FootballRuleBaseModel, FootballKaggle5thPlaceModel = None, None


@POLICY_REGISTRY.register('IL')
class ILPolicy(Policy):
    r"""
    Overview:
        Policy class of Imitation learning algorithm
    Interface:
        __init__, set_setting, __repr__, state_dict_handle
    Property:
        learn_mode, collect_mode, eval_mode
    """
    config = dict(
        type='IL',
        cuda=True,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=False,
        priority=False,
        model=dict(),
        learn=dict(
            multi_gpu=False,
            # (int) collect n_episode data, train model n_iteration time
            update_per_collect=20,
            # (int) the number of data for a train iteration
            batch_size=64,
            # (float) gradient-descent step size
            learning_rate=0.0002,
            # (float) weight decay of optimizer
            weight_decay=0.0,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration time
            n_sample=128,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=800, ), ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
                # (int) max use count of data, if count is bigger than this value,
                # the data will be removed from buffer
                max_reuse=10,
            ),
            command=dict(),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init optimizers, algorithm config, main and target models.
        """
        # actor and critic optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=0.0001)

        # main and target models
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.train()
        self._learn_model.reset()

        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        data = default_collate(data, cat_1dim=False)
        data['done'] = None
        if self._use_cuda:
            data = to_device(data, self._device)
        loss_dict = {}
        # ====================
        # imitation learn forward
        # ====================
        obs = data.get('obs')
        action = data.get('action')
        logit = data.get('logit')
        model_action_logit = self._learn_model.forward(obs['processed_obs'])['logit']
        supervised_loss = nn.MSELoss(reduction='none')(model_action_logit, logit).mean()
        self._optimizer.zero_grad()
        supervised_loss.backward()
        self._optimizer.step()
        loss_dict['supervised_loss'] = supervised_loss
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            **loss_dict,
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect model.
        """
        self._collect_model = model_wrap(FootballKaggle5thPlaceModel(), wrapper_name='base')
        self._gamma = self._cfg.collect.discount_factor
        self._collect_model.eval()
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        with torch.no_grad():
            output = self._collect_model.forward(default_decollate(data['obs']['raw_obs']))
        if self._use_cuda:
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
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, origin_data: deque) -> Union[None, List[Any]]:
        datas = []
        pre_rew = 0
        for i in range(len(origin_data) - 1, -1, -1):
            data = {}
            data['obs'] = origin_data[i]['obs']
            data['action'] = origin_data[i]['action']
            cur_rew = origin_data[i]['reward']
            pre_rew = cur_rew + (pre_rew * self._gamma)
            # sampel uniformly
            data['priority'] = 1
            data['logit'] = origin_data[i]['logit']
            datas.append(data)
        return datas

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.train()
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        with torch.no_grad():
            output = self._eval_model.forward(data['obs']['processed_obs'])
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    # TODO different collect model and learn model
    def default_model(self) -> Tuple[str, List[str]]:
        return 'football_iql', ['app_zoo.gfootball.model.iql.iql_network']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return ['cur_lr', 'supervised_loss']
