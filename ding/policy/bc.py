import math
import torch
import torch.nn as nn
import logging
from torch.optim import Adam, AdamW
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
from easydict import EasyDict
from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device, to_list
from ding.utils import EasyTimer
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import get_nstep_return_data, get_train_sample
from ding.utils import POLICY_REGISTRY
from ding.torch_utils.loss.cross_entropy_loss import LabelSmoothCELoss


@POLICY_REGISTRY.register('bc')
class BehaviourCloningPolicy(Policy):

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.continuous:
            return 'continuous_bc', ['ding.model.template.bc']
        else:
            return 'discrete_bc', ['ding.model.template.bc']

    config = dict(
        type='bc',
        cuda=False,
        on_policy=False,
        continuous=False,
        action_shape=19,
        learn=dict(
            multi_gpu=False,
            update_per_collect=1,
            batch_size=32,
            learning_rate=1e-5,
            weight_decay=None,
            ce_label_smooth=False,
            show_accuracy=False,
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, )),
    )

    def _init_learn(self):
        if self._cfg.learn.weight_decay is None:
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
            )
        else:
            self._optimizer = AdamW(
                self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self._cfg.learn.weight_decay
            )

        self._timer = EasyTimer(cuda=True)
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

        if self._cfg.continuous:
            if self._cfg.loss_type == 'l1_loss':
                self._loss = nn.L1Loss()
            elif self._cfg.loss_type == 'mse_loss':
                self._loss = nn.MSELoss()
            else:
                raise KeyError
        else:
            if not self._cfg.learn.ce_label_smooth:
                self._loss = nn.CrossEntropyLoss()
            else:
                self._loss = LabelSmoothCELoss(0.1)

            if self._cfg.learn.show_accuracy:
                # accuracy statistics for debugging in discrete action space env, e.g. for gfootball
                self.total_accuracy_in_dataset = []
                self.action_accuracy_in_dataset = {k: [] for k in range(self._cfg.action_shape)}

    def _forward_learn(self, data):
        if not isinstance(data, dict):
            data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        with self._timer:
            if self.cfg.eval.evaluator.cfg_type == 'MetricSerialEvaluatorDict':
                obs, action = data
            else:
                obs, action = data['obs'], data['action'].squeeze()
            if self._cfg.continuous:
                mu = self._eval_model.forward(data['obs'])['action']
                loss = self._loss(mu, action)
            else:
                a_logit = self._learn_model.forward(obs)
                loss = self._loss(a_logit['logit'], action)

                if self._cfg.learn.show_accuracy:
                    # Calculate the overall accuracy and the accuracy of each class
                    total_accuracy = (a_logit['action'] == action.view(-1)).float().mean()
                    self.total_accuracy_in_dataset.append(total_accuracy)
                    logging.info(f'the total accuracy in current train mini-batch is: {total_accuracy.item()}')
                    for action_unique in to_list(torch.unique(action)):
                        action_index = (action == action_unique).nonzero(as_tuple=True)[0]
                        action_accuracy = (a_logit['action'][action_index] == action.view(-1)[action_index]
                                           ).float().mean()
                        if math.isnan(action_accuracy):
                            action_accuracy = 0.0
                        self.action_accuracy_in_dataset[action_unique].append(action_accuracy)
                        logging.info(
                            f'the accuracy of action {action_unique} in current train mini-batch is: '
                            f'{action_accuracy.item()}, '
                            f'(nan means the action does not appear in the mini-batch)'
                        )

        forward_time = self._timer.value
        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
        backward_time = self._timer.value
        with self._timer:
            if self._cfg.learn.multi_gpu:
                self.sync_gradients(self._learn_model)
        sync_time = self._timer.value
        self._optimizer.step()
        cur_lr = [param_group['lr'] for param_group in self._optimizer.param_groups]
        cur_lr = sum(cur_lr) / len(cur_lr)
        return {
            'cur_lr': cur_lr,
            'total_loss': loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'sync_time': sync_time,
        }

    def _monitor_vars_learn(self):
        return ['cur_lr', 'total_loss', 'forward_time', 'backward_time', 'sync_time']

    def _init_eval(self):
        if self._cfg.continuous:
            self._eval_model = model_wrap(self._model, wrapper_name='base')
        else:
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data):
        gfootball_flag = False
        tensor_input = isinstance(data, torch.Tensor)
        if tensor_input:
            data = default_collate(list(data))
        else:
            data_id = list(data.keys())
            if data_id == ['processed_obs', 'raw_obs']:
                # for gfootball
                gfootball_flag = True
                data = {0: data}
                data_id = list(data.keys())
            data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        if tensor_input or gfootball_flag:
            return output
        else:
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample
        """
        self._unroll_len = self._cfg.collect.unroll_len
        if self._cfg.continuous:
            self._collect_model = model_wrap(self._model, wrapper_name='base')
        else:
            self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            if self._cfg.continuous:
                output = self._collect_model.forward(data)
            else:
                output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        data = get_nstep_return_data(data, 1, 1)
        return get_train_sample(data, self._unroll_len)
