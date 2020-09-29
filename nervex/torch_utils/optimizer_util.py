import torch
import math
from torch.optim import Adam
from .grad_clip import GradClip, build_grad_clip
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        self._weight_decay = weight_decay
        super(AdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                param.data = param.data.add(-self._weight_decay * group['lr'], param.data)
        return super(AdamW, self).step(closure=closure)


class NervexOptim(Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        optim_type='adam',
        grad_clip_type=None,
        clip_value=None,
        clip_coef=5,
        clip_norm_type=2.0,
        grad_norm_type=None,
        grad_ignore_type=None,
    ):

        self._support_type = {
            'optim': ['adam', 'adamw'],
            'grad_clip': [None, 'clip_const', 'clip_value', 'clip_norm'],
            'grad_norm': [None],
            'grad_ignore': [None],
        }

        assert optim_type in self._support_type['optim']
        assert grad_clip_type in self._support_type['grad_clip']
        assert grad_norm_type in self._support_type['grad_norm']
        assert grad_ignore_type in self._support_type['grad_ignore']
        if not grad_clip_type:
            assert clip_value

        self._optim_type = optim_type
        self._grad_clip_type = grad_clip_type
        self._grad_norm_type = grad_norm_type
        self._grad_ignore_type = grad_ignore_type
        self._clip_value = clip_value
        self._clip_norm_type = clip_norm_type
        self._clip_coef = clip_coef

        if self._optim_type == 'adamw':
            self._weight_decay = weight_decay
            super(NervexOptim, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)
        elif self._optim_type == 'adam':
            super(NervexOptim, self).__init__(
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
            )
        else:
            raise NotImplementedError(
                "optimizer type {} is not implemented, support type is {}".format(
                    self._optim_type, self._support_type['optim']
                )
            )

    def step(self, closure=None):
        #clipping
        new_params = []
        for group in self.param_groups:
            new_params += [t for t in group['params'] if t.requires_grad and t.grad is not None]
        if self._grad_clip_type == 'clip_const':
            clip_grad_value_(new_params, self._clip_value)

        if self._grad_clip_type == 'clip_norm':
            clip_grad_norm_(new_params, self._clip_value, self._clip_norm_type)

        if self._grad_clip_type == 'clip_value':
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if len(state) == 0:
                        # Exponential moving average of squared gradient values
                        state['clip_exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)
                        state['step'] = 0
                    #should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2**state['step']
                    state['clip_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] >= 100:  # initial value is inaccurate
                        flag = grad.abs(
                        ) > (state['clip_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) * self._clip_coef
                        grad.mul_(~flag).add_(
                            ((state['clip_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) *
                             self._clip_coef).mul_(flag)
                        )

        #Adam optim type
        if self._optim_type == 'adamw':
            for group in self.param_groups:
                for p in group['params']:
                    p.data = p.data.add(-self._weight_decay * group['lr'], p.data)
            return super().step(closure=closure)
        elif self._optim_type == 'adam':
            return super().step(closure=closure)
