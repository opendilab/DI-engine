import math
import torch
import torch.nn as nn
import copy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
from easydict import EasyDict
from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils import EasyTimer
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import q_nstep_td_data, q_nstep_sql_td_error, get_nstep_return_data, get_train_sample
from ding.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('bco')
class BCOPolicy(Policy):
    config = dict(
        type='bco',
        on_policy=False,
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            #n_sample=8,  # collect 8 samples and put them in collector
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
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
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self):
        self._optimizer = SGD(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            weight_decay=self._cfg.learn.weight_decay,
        )
        self._timer = EasyTimer(cuda=True)

        def lr_scheduler_fn(epoch):
            if epoch <= self._cfg.learn.warmup_epoch:
                return self._cfg.learn.warmup_lr / self._cfg.learn.learning_rate
            else:
                ratio = (epoch - self._cfg.learn.warmup_epoch) // self._cfg.learn.decay_epoch
                return math.pow(self._cfg.learn.decay_rate, ratio)

        self._lr_scheduler = LambdaLR(self._optimizer, lr_scheduler_fn)

        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

        self._ce_loss = nn.CrossEntropyLoss()

    def _forward_learn(self, data):
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        # ====================
        # forward
        # ====================
        with self._timer:
            obs, action = data['obs'], data['action']
            a_logit = self._learn_model.forward(obs)
            loss = self._ce_loss(a_logit['logit'], action)
        forward_time = self._timer.value
        # ====================
        # update
        # ====================
        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
        backward_time = self._timer.value

        with self._timer:
            if self._cfg.learn.multi_gpu:
                self.sync_gradients(self._learn_model)
        sync_time = self._timer.value
        self._optimizer.step()
        cur_lr = self._lr_scheduler.get_last_lr()[0]
        return {
            'cur_lr': cur_lr,
            'total_loss': loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'sync_time': sync_time,
            '[histogram]logit_distribution':a_logit['logit'],
        }

    def _monitor_vars_learn(self):
        return ['[histogram]logit_distribution','cur_lr', 'total_loss', 'forward_time', 'backward_time', 'sync_time']

    def _init_eval(self):
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data):
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
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
        #self._gamma = self._cfg.discount_factor  # necessary for parallel
        #self._nstep = self._cfg.nstep  # necessary for parallel
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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'bc', ['ding.model.template.bc']
