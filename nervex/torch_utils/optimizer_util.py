import torch
import math
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch._six import inf

from .grad_clip import GradClip, build_grad_clip


def grad_ignore_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.zero_()
    return total_norm


def grad_ignore_value(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    flag = False
    for p in filter(lambda p: p.grad is not None, parameters):
        if p.grad.data > clip_value or p.grad.data < -clip_value:
            flag = True
    if flag:
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.zero_()


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
        clip_momentum_timestep=100,
        grad_norm_type=None,
        grad_ignore_type=None,
        ignore_value=None,
        ignore_coef=5,
        ignore_norm_type=2.0,
        ignore_momentum_timestep=100,
    ):

        self._support_type = {
            'optim': ['adam', 'adamw'],
            'grad_clip': [None, 'clip_momentum', 'clip_value', 'clip_norm'],
            'grad_norm': [None],
            'grad_ignore': [None, 'ignore_momentum', 'ignore_value', 'ignore_norm'],
        }

        assert optim_type in self._support_type['optim']
        assert grad_clip_type in self._support_type['grad_clip']
        assert grad_norm_type in self._support_type['grad_norm']
        assert grad_ignore_type in self._support_type['grad_ignore']
        if grad_clip_type:
            assert clip_value is not None
        if grad_ignore_type:
            assert ignore_value is not None

        self._optim_type = optim_type
        self._grad_clip_type = grad_clip_type
        self._grad_norm_type = grad_norm_type
        self._grad_ignore_type = grad_ignore_type
        self._clip_value = clip_value
        self._clip_norm_type = clip_norm_type
        self._clip_coef = clip_coef
        self._ignore_value = ignore_value
        self._ignore_norm_type = ignore_norm_type
        self._ignore_coef = ignore_coef
        self._clip_momentum_timestep = clip_momentum_timestep
        self._ignore_momentum_timestep = ignore_momentum_timestep

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
        # clipping
        # new_params = []
        # for group in self.param_groups:
        #     new_params += [t for t in group['params'] if t.requires_grad and t.grad is not None]
        new_params = [
            t for group in self.param_groups for t in group['params'] if t.requires_grad and t.grad is not None
        ]
        if self._grad_clip_type == 'clip_value':
            clip_grad_value_(new_params, self._clip_value)
        elif self._grad_clip_type == 'clip_norm':
            clip_grad_norm_(new_params, self._clip_value, self._clip_norm_type)
        elif self._grad_clip_type == 'clip_momentum':
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if len(state) == 0:
                        # this is a big if
                        state['clip_exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)

                        #state['ignore_exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)

                        # others
                        state['step'] = 0
                        #TODO
                        #wait torch upgrad to 1.4, 1.3.1 didn't support memory format;ate['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    #should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['clip_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] >= self._clip_momentum_timestep:  # initial value is inaccurate
                        flag = grad.abs(
                        ) > (state['clip_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) * self._clip_coef
                        grad.mul_(~flag).add_(
                            ((state['clip_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) *
                             self._clip_coef).mul_(flag)
                        )

        if self._grad_ignore_type == 'ignore_value':
            grad_ignore_value(new_params, self._ignore_value)
        elif self._grad_ignore_type == 'ignore_norm':
            grad_ignore_norm(new_params, self._ignore_value, self._ignore_norm_type)
        elif self._grad_ignore_type == 'ignore_momentum':
            flag = False
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if len(state) == 0:
                        # this is a big if
                        state['clip_exp_avg_sq'] = torch.zeros_like(p.data)

                        #state['ignore_exp_avg_sq'] = torch.zeros_like(p.data)

                        # others
                        state['step'] = 0
                        #TODO
                        #wait torch upgrad to 1.4, 1.3.1 didn't support memory format;
                        #state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        state['exp_avg'] = torch.zeros_like(p.data)

                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    #should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    # state['ignore_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    state['clip_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] >= self._ignore_momentum_timestep:  # initial value is inaccurate
                        if grad.abs() > (state['clip_exp_avg_sq'].sqrt() /
                                         math.sqrt(bias_correction2)) * self._ignore_coef:
                            flag = True
                            break
                else:
                    continue
                break

            if flag:
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.grad.zero_()

        #Adam optim type
        if self._optim_type == 'adamw':
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data = p.data.add(-self._weight_decay * group['lr'], p.data)
            return super().step(closure=closure)
        elif self._optim_type == 'adam':
            return super().step(closure=closure)
