import math
from typing import List, Dict, Any, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from easydict import EasyDict

from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils import EasyTimer
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import get_nstep_return_data, get_train_sample
from ding.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('pc_mcts')
class ProcedureCloningPolicyMCTS(Policy):
    config = dict(
        type='pc_mcts',
        cuda=True,
        on_policy=False,
        continuous=False,
        learn=dict(
            multi_gpu=False,
            update_per_collect=1,
            batch_size=32,
            learning_rate=1e-5,
            lr_decay=False,
            decay_epoch=30,
            decay_rate=0.1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            momentum=0.9,
            weight_decay=1e-4,
            ce_label_smooth=False,
            show_accuracy=False,
            tanh_mask=False,  # if actions always converge to 1 or -1, use this.
        ),
        collect=dict(
            unroll_len=1,
            noise=False,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, )),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'pc_mcts', ['ding.model.template.procedure_cloning']

    def _init_learn(self):
        assert self._cfg.learn.optimizer in ['SGD', 'Adam']
        if self._cfg.learn.optimizer == 'SGD':
            self._optimizer = SGD(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                weight_decay=self._cfg.learn.weight_decay,
                momentum=self._cfg.learn.momentum
            )
        elif self._cfg.learn.optimizer == 'Adam':
            if self._cfg.learn.weight_decay is None:
                self._optimizer = Adam(
                    self._model.parameters(),
                    lr=self._cfg.learn.learning_rate,
                )
            else:
                self._optimizer = AdamW(
                    self._model.parameters(),
                    lr=self._cfg.learn.learning_rate,
                    weight_decay=self._cfg.learn.weight_decay
                )
        if self._cfg.learn.lr_decay:

            def lr_scheduler_fn(epoch):
                if epoch <= self._cfg.learn.warmup_epoch:
                    return self._cfg.learn.warmup_lr / self._cfg.learn.learning_rate
                else:
                    ratio = (epoch - self._cfg.learn.warmup_epoch) // self._cfg.learn.decay_epoch
                    return math.pow(self._cfg.learn.decay_rate, ratio)

            self._lr_scheduler = LambdaLR(self._optimizer, lr_scheduler_fn)
        self._timer = EasyTimer(cuda=True)
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

        self._hidden_state_loss = nn.MSELoss()
        self._action_loss = nn.CrossEntropyLoss()

    def _forward_learn(self, data):
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        with self._timer:
            obs, hidden_states, action = data['obs'], data['hidden_states'], data['action']
            zero_hidden_len = len(hidden_states) > 0
            if zero_hidden_len:
                hidden_states = torch.stack(hidden_states, dim=1).float()
            else:
                hidden_states = to_device(torch.empty(obs.shape[0], 0, *self._learn_model.hidden_shape), self._device)
            pred_hidden_states, pred_action, target_hidden_states = self._learn_model.forward(obs, hidden_states)
            if zero_hidden_len:
                hidden_state_loss = 0
            else:
                hidden_state_loss = self._hidden_state_loss(pred_hidden_states, target_hidden_states)
            action_loss = self._action_loss(pred_action, action)
            loss = hidden_state_loss + action_loss
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
            'hidden_state_loss': hidden_state_loss.item(),
            'action_loss': action_loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'sync_time': sync_time,
        }

    def _monitor_vars_learn(self):
        return ['cur_lr', 'total_loss', 'hidden_state_loss', 'action_loss',
                'forward_time', 'backward_time', 'sync_time']

    def _init_eval(self):
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data):
        data_id = list(data.keys())
        values = list(data.values())
        data = [{'obs': v['observation']} for v in values]
        data = default_collate(data)

        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward_eval(data['obs'].permute(0, 3, 1, 2) / 255.)
            output = torch.argmax(output, dim=-1)
            if self._cuda:
                output = to_device(output, 'cpu')
            output = {'action': output}
        output = default_decollate(output)
        # TODO why this bug?
        output = [{'action': o['action'].item()} for o in output]
        res = {i: d for i, d in zip(data_id, output)}
        return res

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        pass

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
