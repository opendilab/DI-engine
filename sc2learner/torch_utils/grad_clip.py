import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def build_grad_clip(cfg):
    clip_type = cfg.train.grad_clip_type
    clip_value = cfg.train.grad_clip_value
    return GradClip(clip_value, clip_type)


class GradClip(object):
    def __init__(self, clip_value, clip_type, norm_type=2):
        assert (clip_type in ['max_norm', 'clip_value'])
        self.norm_type = norm_type
        self.clip_value = clip_value
        self.clip_type = clip_type

    def apply(self, parameters, value=None):
        if value is None:
            v = self.clip_value
        else:
            v = value
        new_params = [t for t in parameters if t.requires_grad and t.grad is not None]
        if self.clip_type == 'max_norm':
            clip_grad_norm_(new_params, v, self.norm_type)
        elif self.clip_type == 'clip_value':
            clip_grad_value_(new_params, v)
