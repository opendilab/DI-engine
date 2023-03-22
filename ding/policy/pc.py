import math
from typing import List, Dict, Any, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR

from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils import EasyTimer
from ding.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('pc_bfs')
class ProcedureCloningBFSPolicy(Policy):

    def default_model(self) -> Tuple[str, List[str]]:
        return 'pc_bfs', ['ding.model.template.procedure_cloning']

    config = dict(
        type='pc',
        cuda=False,
        on_policy=False,
        continuous=False,
        max_bfs_steps=100,
        learn=dict(
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
        other=dict(replay_buffer=dict(replay_buffer_size=10000)),
    )

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
        self._max_bfs_steps = self._cfg.max_bfs_steps
        self._maze_size = self._cfg.maze_size
        self._num_actions = self._cfg.num_actions

        self._loss = nn.CrossEntropyLoss()

    def process_states(self, observations, maze_maps):
        """Returns [B, W, W, 3] binary values. Channels are (wall; goal; obs)"""
        loc = torch.nn.functional.one_hot(
            (observations[:, 0] * self._maze_size + observations[:, 1]).long(),
            self._maze_size * self._maze_size,
        ).long()
        loc = torch.reshape(loc, [observations.shape[0], self._maze_size, self._maze_size])
        states = torch.cat([maze_maps, loc], dim=-1).long()
        return states

    def _forward_learn(self, data):
        if self._cuda:
            collated_data = to_device(data, self._device)
        else:
            collated_data = data
        observations = collated_data['obs'],
        bfs_input_maps, bfs_output_maps = collated_data['bfs_in'].long(), collated_data['bfs_out'].long()
        states = observations
        bfs_input_onehot = torch.nn.functional.one_hot(bfs_input_maps, self._num_actions + 1).float()

        bfs_states = torch.cat([
            states,
            bfs_input_onehot,
        ], dim=-1)
        logits = self._model(bfs_states)['logit']
        logits = logits.flatten(0, -2)
        labels = bfs_output_maps.flatten(0, -1)

        loss = self._loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum((preds == labels)) / preds.shape[0]

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        pred_loss = loss.item()

        cur_lr = [param_group['lr'] for param_group in self._optimizer.param_groups]
        cur_lr = sum(cur_lr) / len(cur_lr)
        return {'cur_lr': cur_lr, 'total_loss': pred_loss, 'acc': acc}

    def _monitor_vars_learn(self):
        return ['cur_lr', 'total_loss', 'acc']

    def _init_eval(self):
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data):
        if self._cuda:
            data = to_device(data, self._device)
        max_len = self._max_bfs_steps
        data_id = list(data.keys())
        output = {}

        for ii in data_id:
            states = data[ii].unsqueeze(0)
            bfs_input_maps = self._num_actions * torch.ones([1, self._maze_size, self._maze_size]).long()
            if self._cuda:
                bfs_input_maps = to_device(bfs_input_maps, self._device)
            xy = torch.where(states[:, :, :, -1] == 1)
            observation = (xy[1][0].item(), xy[2][0].item())

            i = 0
            while bfs_input_maps[0, observation[0], observation[1]].item() == self._num_actions and i < max_len:
                bfs_input_onehot = torch.nn.functional.one_hot(bfs_input_maps, self._num_actions + 1).long()

                bfs_states = torch.cat([
                    states,
                    bfs_input_onehot,
                ], dim=-1)
                logits = self._model(bfs_states)['logit']
                bfs_input_maps = torch.argmax(logits, dim=-1)
                i += 1
            output[ii] = bfs_input_maps[0, observation[0], observation[1]]
            if self._cuda:
                output[ii] = {'action': to_device(output[ii], 'cpu'), 'info': {}}
            if output[ii]['action'].item() == self._num_actions:
                output[ii]['action'] = torch.randint(low=0, high=self._num_actions, size=[1])[0]
        return output

    def _init_collect(self) -> None:
        raise NotImplementedError

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        raise NotImplementedError

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
