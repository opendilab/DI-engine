from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import torch.nn as nn

from ding.torch_utils import Adam, to_device
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
FootballKaggle5thPlaceModel = None


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
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            multi_gpu=False,
            # (int) collect n_episode data, train model n_iteration time
            update_per_collect=20,
            # (int) the number of data for a train iteration
            batch_size=64,
            # (float) gradient-descent step size
            learning_rate=0.0002,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration time
            # n_sample=128,
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
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

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
        if self._cuda:
            data = to_device(data, self._device)
        loss_dict = {}
        # ====================
        # imitation learn forward
        # ====================
        obs = data.get('obs')
        logit = data.get('logit')
        assert isinstance(obs['processed_obs'], torch.Tensor), obs['processed_obs']
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
            Init traj and unroll length, collect model.
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
        with torch.no_grad():
            output = self._collect_model.forward(default_decollate(data['obs']['raw_obs']))
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
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, origin_data: list) -> Union[None, List[Any]]:
        datas = []
        pre_rew = 0
        for i in range(len(origin_data) - 1, -1, -1):
            data = {}
            data['obs'] = origin_data[i]['obs']
            data['action'] = origin_data[i]['action']
            cur_rew = origin_data[i]['reward']
            pre_rew = cur_rew + (pre_rew * self._gamma)
            # sample uniformly
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
        with torch.no_grad():
            output = self._eval_model.forward(data['obs']['processed_obs'])
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    # TODO different collect model and learn model
    def default_model(self) -> Tuple[str, List[str]]:
        return 'football_iql', ['dizoo.gfootball.model.iql.iql_network']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return ['cur_lr', 'supervised_loss']
