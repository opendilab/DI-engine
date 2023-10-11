import torch
import math
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from typing import Union, Iterable, Tuple, Callable, List
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

inf = math.inf


def calculate_grad_norm(model: torch.nn.Module, norm_type=2) -> float:
    r"""
    Overview:
        calculate grad norm of the parameters whose grad norms are not None in the model.
    Arguments:
        - model: torch.nn.Module
        - norm_type (:obj:`int` or `inf`)
    """
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    if parameters == []:
        parameters = 0
        return 0
    if norm_type == 'inf':
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        return float(total_norm)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return float(total_norm)


def calculate_grad_norm_without_bias_two_norm(model: torch.nn.Module) -> float:
    r"""
    Overview:
        calculate grad norm of the parameters whose grad norms are not None in the model.
    Arguments:
        - model: torch.nn.Module
    """
    _list = []
    for name, param in model.named_parameters():
        if 'bias' not in name and param.requires_grad:
            if param.grad is None:
                return 0
            _list.append(param.grad.data.norm(2).item() ** 2)
    return float(sum(_list) ** (1. / 2))


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
        val = p.grad.data.abs().max()
        if val >= clip_value:
            flag = True
            break
    if flag:
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.zero_()


class Adam(torch.optim.Adam):
    r"""
    Overview:
        Rewrited Adam optimizer to support more features.
    Interface:
        __init__, step
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        optim_type: str = 'adam',
        grad_clip_type: str = None,
        clip_value: Union[float, None] = None,
        clip_coef: float = 5,
        clip_norm_type: float = 2.0,
        clip_momentum_timestep: int = 100,
        grad_norm_type: str = None,
        grad_ignore_type: str = None,
        ignore_value: Union[float, None] = None,
        ignore_coef: float = 5,
        ignore_norm_type: float = 2.0,
        ignore_momentum_timestep: int = 100,
    ):
        r"""
        Overview:
            init method of refactored Adam class
        Arguments:
            - params (:obj:`iterable`):  – an iterable of torch.Tensor s or dict s. \
                Specifies what Tensors should be optimized
            - lr (:obj:`float`): learning rate, default set to 1e-3
            - betas (:obj:`Tuple[float, float]`): coefficients used for computing running averages of gradient and its\
                square, default set to (0.9, 0.999))
            - eps (:obj:`float`): term added to the denominator to improve numerical stability, default set to 1e-8
            - weight_decay (:obj:`float`): weight decay coefficient, deault set to 0
            - amsgrad (:obj:`bool`): whether to use the AMSGrad variant of this algorithm from the paper\
                On the Convergence of Adam and Beyond <https://arxiv.org/abs/1904.09237>
            - optim_type (:obj:str): support ["adam", "adamw"]
            - grad_clip_type (:obj:`str`): support [None, 'clip_momentum', 'clip_value', 'clip_norm', \
                'clip_momentum_norm']
            - clip_value (:obj:`float`): the value to start clipping
            - clip_coef (:obj:`float`): the cliping coefficient
            - clip_norm_type (:obj:`float`): 2.0 means use norm2 to clip
            - clip_momentum_timestep (:obj:`int`): after how many step should we start the momentum clipping
            - grad_ignore_type (:obj:`str`): support [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', \
                'ignore_momentum_norm']
            - ignore_value (:obj:`float`): the value to start ignoring
            - ignore_coef (:obj:`float`): the ignoreing coefficient
            - ignore_norm_type (:obj:`float`): 2.0 means use norm2 to ignore
            - ignore_momentum_timestep (:obj:`int`): after how many step should we start the momentum ignoring

        """

        self._support_type = {
            'optim': ['adam', 'adamw'],
            'grad_clip': [None, 'clip_momentum', 'clip_value', 'clip_norm', 'clip_momentum_norm'],
            'grad_norm': [None],
            'grad_ignore': [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', 'ignore_momentum_norm'],
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
            super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)
        elif self._optim_type == 'adam':
            super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        else:
            raise NotImplementedError(
                "optimizer type {} is not implemented, support type is {}".format(
                    self._optim_type, self._support_type['optim']
                )
            )

    def _state_init(self, p, amsgrad):
        state = self.state[p]
        state['thre_exp_avg_sq'] = torch.zeros_like(p.data, device=p.data.device)
        # others
        if torch.__version__ < "1.12.0":
            state['step'] = 0
            # TODO
            # wait torch upgrad to 1.4, 1.3.1 didn't support memory format state['step'] = 0
        else:
            state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                if self.defaults['capturable'] else torch.tensor(0.)

        state['exp_avg'] = torch.zeros_like(p.data)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p.data)
        if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure: Union[Callable, None] = None):
        r"""
        Overview:
            Performs a single optimization step
        Arguments:
            - closure (:obj:`callable`): A closure that reevaluates the model and returns the loss, default set to None
        """
        # clipping
        new_params = [
            t for group in self.param_groups for t in group['params'] if t.requires_grad and t.grad is not None
        ]
        if self._grad_clip_type == 'clip_value':
            clip_grad_value_(new_params, self._clip_value)
        elif self._grad_clip_type == 'clip_norm':
            clip_grad_norm_(new_params, self._clip_value, self._clip_norm_type)
        elif self._grad_clip_type == 'clip_momentum':
            '''
            This is the implimentation mimic the clip used in OPENAI, quote:
                'Gradients are additionally clipped per parameter to be within between ±5√v
                 where v is the running estimate of the second moment of the (unclipped) gradient'
            '''
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['amsgrad'])
                    grad = p.grad.data
                    # should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['thre_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] >= self._clip_momentum_timestep:  # initial value is inaccurate
                        flag = grad.abs(
                        ) > (state['thre_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) * self._clip_coef
                        grad.mul_(~flag).add_(
                            ((state['thre_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) *
                             self._clip_coef).mul_(flag)
                        )
        elif self._grad_clip_type == 'clip_momentum_norm':
            # might have multi param_group, we should calculate each group differently.
            for group in self.param_groups:
                total_norm = 0
                total_momentum_norm = 0
                step = inf
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['amsgrad'])
                    grad = p.grad.data
                    # should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['thre_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # sum total_norm
                    param_norm = grad.norm(self._clip_norm_type)
                    total_norm += param_norm.item() ** self._clip_norm_type

                    # sum momentum_norm
                    momentum = ((state['thre_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) *
                                self._clip_coef).norm(self._clip_norm_type)
                    total_momentum_norm += momentum.item() ** self._clip_norm_type
                    step = min(step, state['step'])
                if step > self._clip_momentum_timestep:
                    total_norm = total_norm ** (1. / self._clip_norm_type)
                    total_momentum_norm = total_momentum_norm ** (1. / self._clip_norm_type)
                    clip_coef = total_momentum_norm / (total_norm + 1e-6)
                    if clip_coef < 1:
                        for p in group['params']:
                            p.grad.data.mul_(clip_coef)

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
                    if len(state) == 0:
                        self._state_init(p, group['amsgrad'])
                    grad = p.grad.data
                    # should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['thre_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] >= self._ignore_momentum_timestep:  # initial value is inaccurate
                        if grad.abs() > (state['thre_exp_avg_sq'].sqrt() /
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
        elif self._grad_ignore_type == 'ignore_momentum_norm':
            # might have multi param_group, we should calculate each group differently.
            step = inf
            for group in self.param_groups:
                total_norm = 0
                total_momentum_norm = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['amsgrad'])
                    grad = p.grad.data
                    # should we use same beta group?
                    beta1, beta2 = group['betas']
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['thre_exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # sum total_norm
                    param_norm = grad.norm(self._ignore_norm_type)
                    total_norm += param_norm.item() ** self._ignore_norm_type

                    # sum momentum_norm
                    momentum = ((state['thre_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) *
                                self._ignore_coef).norm(self._ignore_norm_type)
                    total_momentum_norm += momentum.item() ** self._ignore_norm_type
                    step = min(step, state['step'])

                if step > self._ignore_momentum_timestep:
                    total_norm = total_norm ** (1. / self._ignore_norm_type)
                    total_momentum_norm = total_momentum_norm ** (1. / self._ignore_norm_type)
                    ignore_coef = total_momentum_norm / (total_norm + 1e-6)
                    if ignore_coef < 1:
                        for p in group['params']:
                            p.grad.zero_()

        # Adam optim type
        if self._optim_type == 'adamw':
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data = p.data.add(-self._weight_decay * group['lr'], p.data)
            return super().step(closure=closure)
        elif self._optim_type == 'adam':
            return super().step(closure=closure)

    def get_grad(self) -> float:
        total_norm = 0.
        params = [t for group in self.param_groups for t in group['params'] if t.requires_grad and t.grad is not None]
        for p in params:
            param_norm = p.grad.data.norm(self._clip_norm_type)
            total_norm += param_norm.item() ** self._clip_norm_type
        return total_norm


class RMSprop(torch.optim.RMSprop):
    r"""
    Overview:
        Rewrited RMSprop optimizer to support more features.
    Interface:
        __init__, step
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        grad_clip_type: str = None,
        clip_value: Union[float, None] = None,
        clip_coef: float = 5,
        clip_norm_type: float = 2.0,
        clip_momentum_timestep: int = 100,
        grad_norm_type: str = None,
        grad_ignore_type: str = None,
        ignore_value: Union[float, None] = None,
        ignore_coef: float = 5,
        ignore_norm_type: float = 2.0,
        ignore_momentum_timestep: int = 100,
    ):
        r"""
        Overview:
            init method of refactored Adam class
        Arguments:
            - params (:obj:`iterable`):  – an iterable of torch.Tensor s or dict s. \
                Specifies what Tensors should be optimized
            - lr (:obj:`float`): learning rate, default set to 1e-3
            - alpha (:obj:`float`): smoothing constant, default set to 0.99
            - eps (:obj:`float`): term added to the denominator to improve numerical stability, default set to 1e-8
            - weight_decay (:obj:`float`): weight decay coefficient, deault set to 0
            - centred (:obj:`bool`): if True, compute the centered RMSprop, \
                the gradient is normalized by an estimation of its variance
            - grad_clip_type (:obj:`str`): support [None, 'clip_momentum', 'clip_value', 'clip_norm', \
                'clip_momentum_norm']
            - clip_value (:obj:`float`): the value to start clipping
            - clip_coef (:obj:`float`): the cliping coefficient
            - clip_norm_type (:obj:`float`): 2.0 means use norm2 to clip
            - clip_momentum_timestep (:obj:`int`): after how many step should we start the momentum clipping
            - grad_ignore_type (:obj:`str`): support [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', \
                'ignore_momentum_norm']
            - ignore_value (:obj:`float`): the value to start ignoring
            - ignore_coef (:obj:`float`): the ignoreing coefficient
            - ignore_norm_type (:obj:`float`): 2.0 means use norm2 to ignore
            - ignore_momentum_timestep (:obj:`int`): after how many step should we start the momentum ignoring
        """

        self._support_type = {
            'grad_clip': [None, 'clip_momentum', 'clip_value', 'clip_norm', 'clip_momentum_norm'],
            'grad_norm': [None],
            'grad_ignore': [None, 'ignore_momentum', 'ignore_value', 'ignore_norm', 'ignore_momentum_norm'],
        }

        assert grad_clip_type in self._support_type['grad_clip']
        assert grad_norm_type in self._support_type['grad_norm']
        assert grad_ignore_type in self._support_type['grad_ignore']
        if grad_clip_type:
            assert clip_value is not None
        if grad_ignore_type:
            assert ignore_value is not None

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

        super(RMSprop, self).__init__(
            params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered
        )

    def _state_init(self, p, momentum, centered):
        state = self.state[p]
        state['step'] = 0
        state['thre_square_avg'] = torch.zeros_like(p.data, device=p.data.device)
        state['square_avg'] = torch.zeros_like(p.data, device=p.data.device)
        if momentum:
            state['momentum_buffer'] = torch.zeros_like(p.data, device=p.data.device)
        if centered:
            state['grad_avg'] = torch.zeros_like(p.data, device=p.data.device)

    def step(self, closure: Union[Callable, None] = None):
        r"""
        Overview:
            Performs a single optimization step
        Arguments:
            - closure (:obj:`callable`): A closure that reevaluates the model and returns the loss, default set to None
        """
        # clipping
        new_params = [
            t for group in self.param_groups for t in group['params'] if t.requires_grad and t.grad is not None
        ]
        if self._grad_clip_type == 'clip_value':
            clip_grad_value_(new_params, self._clip_value)
        elif self._grad_clip_type == 'clip_norm':
            clip_grad_norm_(new_params, self._clip_value, self._clip_norm_type)
        elif self._grad_clip_type == 'clip_momentum':
            '''
                 This implementation mimics the clip used in OPENAI, quote:
                'Gradients are additionally clipped per parameter to be within between ±5√v
                 where v is the running estimate of the second moment of the (unclipped) gradient'
            '''
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['momentum'], group['centered'])
                    grad = p.grad.data
                    # beta1, beta2 = group['betas']
                    alpha = group['alpha']
                    state['thre_square_avg'].mul_(alpha).addcmul_(1 - alpha, grad, grad)
                    if state['step'] >= self._clip_momentum_timestep:  # initial value is inaccurate
                        flag = grad.abs() > state['thre_square_avg'].sqrt() * self._clip_coef
                        grad.mul_(~flag).add_((state['thre_square_avg'].sqrt() * self._clip_coef).mul_(flag))
        elif self._grad_clip_type == 'clip_momentum_norm':
            # might have multi param_group, we should calculate each group differently.
            for group in self.param_groups:
                total_norm = 0
                total_momentum_norm = 0
                step = inf
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['momentum'], group['centered'])
                    grad = p.grad.data
                    alpha = group['alpha']
                    state['thre_square_avg'].mul_(alpha).addcmul_(1 - alpha, grad, grad)
                    # sum total_norm
                    param_norm = grad.norm(self._clip_norm_type)
                    total_norm += param_norm.item() ** self._clip_norm_type

                    # sum momentum_norm
                    momentum = (state['thre_square_avg'].sqrt() * self._clip_coef).norm(self._clip_norm_type)
                    total_momentum_norm += momentum.item() ** self._clip_norm_type
                    step = min(step, state['step'])
                if step > self._clip_momentum_timestep:
                    total_norm = total_norm ** (1. / self._clip_norm_type)
                    total_momentum_norm = total_momentum_norm ** (1. / self._clip_norm_type)
                    clip_coef = total_momentum_norm / (total_norm + 1e-6)
                    if clip_coef < 1:
                        for p in group['params']:
                            p.grad.data.mul_(clip_coef)

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
                    if len(state) == 0:
                        self._state_init(p, group['momentum'], group['centered'])
                    grad = p.grad.data
                    alpha = group['alpha']
                    state['thre_square_avg'].mul_(alpha).addcmul_(1 - alpha, grad, grad)
                    if state['step'] >= self._ignore_momentum_timestep:  # initial value is inaccurate
                        if grad.abs() > state['thre_square_avg'].sqrt() * self._ignore_coef:
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
        elif self._grad_ignore_type == 'ignore_momentum_norm':
            # might have multi param_group, we should calculate each group differently.
            step = inf
            for group in self.param_groups:
                total_norm = 0
                total_momentum_norm = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        self._state_init(p, group['momentum'], group['centered'])
                    grad = p.grad.data
                    alpha = group['alpha']
                    state['thre_square_avg'].mul_(alpha).addcmul_(1 - alpha, grad, grad)
                    # sum total_norm
                    param_norm = grad.norm(self._ignore_norm_type)
                    total_norm += param_norm.item() ** self._ignore_norm_type

                    # sum momentum_norm
                    momentum = (state['thre_square_avg'].sqrt() * self._ignore_coef).norm(self._ignore_norm_type)
                    total_momentum_norm += momentum.item() ** self._ignore_norm_type
                    step = min(step, state['step'])

                if step > self._ignore_momentum_timestep:
                    total_norm = total_norm ** (1. / self._ignore_norm_type)
                    total_momentum_norm = total_momentum_norm ** (1. / self._ignore_norm_type)
                    ignore_coef = total_momentum_norm / (total_norm + 1e-6)
                    if ignore_coef < 1:
                        for p in group['params']:
                            p.grad.zero_()

        return super().step(closure=closure)

    def get_grad(self) -> float:
        total_norm = 0.
        params = [t for group in self.param_groups for t in group['params'] if t.requires_grad and t.grad is not None]
        for p in params:
            param_norm = p.grad.data.norm(self._clip_norm_type)
            total_norm += param_norm.item() ** self._clip_norm_type
        return total_norm


class PCGrad():
    """
    Overview:
        PCGrad optimizer to support multi-task.
        you can view the paper in the following link https://arxiv.org/pdf/2001.06782.pdf
    """

    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        Arguments:
            - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            raise KeyError("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        Returns:
            - grad: a list of the gradient of the parameters
            - shape: a list of the shape of the parameters
            - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific objective
        Returns:
            - grad: a list of the gradient of the parameters
            - shape: a list of the shape of the parameters
            - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


def configure_weight_decay(model: nn.Module, weight_decay: float) -> List:
    r"""
    Overview:
        Separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layer-norm or embedding weights).
    Arguments:
        - model (:obj:`nn.Module`): the given PyTorch model.
        - weight_decay (:obj:`float`): weight decay value for optimizer.
    Returns:
        - optim groups (:obj:`List`): the parameter groups to be set in the latter optimizer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # Because named_modules and named_parameters are recursive
            # we will see the same tensors p many times. But doing it this way
            # allows us to know which parent module any tensor p belongs to.
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    decay = decay - no_decay
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    union_params = decay | no_decay
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]

    return optim_groups
