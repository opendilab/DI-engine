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
        clip_norm_type=2,
        grad_norm_type=None,
        grad_ignore_type=None,
    ):

        self._support_type = {
            'optim': ['adam', 'adamw'],
            'grad_clip': [None, 'value', 'norm'],
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
        new_params = [t for t in self.param_groups if t.requires_grad and t.grad is not None]
        if self._grad_clip_type == 'value':
            clip_grad_value_(new_params, self._clip_value)
        if self._grad_clip_type == 'norm':
            clip_grad_norm_(new_params, self._clip_value, self._clip_norm_type)

        #Adam optim type
        if self._optim_type == 'adamw':
            for group in self.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-self._weight_decay * group['lr'], param.data)
            return super().step(closure=closure)
        elif self._optim_type == 'adam':
            return super().step(closure=closure)
