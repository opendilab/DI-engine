import math
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils import EasyTimer


class ImageClassificationPolicy(Policy):
    config = dict(
        type='image_classification',
        on_policy=False,
    )

    def _init_learn(self):
        self._optimizer = SGD(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            weight_decay=self._cfg.learn.weight_decay,
            momentum=0.9
        )
        self._timer = EasyTimer(cuda=True)

        def lr_scheduler_fn(epoch):
            if epoch <= self._cfg.learn.warmup_epoch:
                return self._cfg.learn.warmup_lr / self._cfg.learn.learning_rate
            else:
                ratio = epoch // self._cfg.learn.decay_epoch
                return math.pow(self._cfg.learn.decay_rate, ratio)

        self._lr_scheduler = LambdaLR(self._optimizer, lr_scheduler_fn)
        self._lr_scheduler.step()
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

        self._ce_loss = nn.CrossEntropyLoss()

    def _forward_learn(self, data):
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()

        with self._timer:
            img, target = data
            logit = self._learn_model.forward(img)
            loss = self._ce_loss(logit, target)
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
        self._eval_model = model_wrap(self._model, 'base')

    def _forward_eval(self, data):
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        return output

    def _init_collect(self):
        pass

    def _forward_collect(self, data):
        pass

    def _process_transition(self):
        pass

    def _get_train_sample(self):
        pass
