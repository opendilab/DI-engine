"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Grad_clip: Clip the gradients.
"""
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def build_grad_clip(cfg):
    r"""
    Overview:
        build the gradient cliper

    Arguments:
        Notes:
            for more detail you can view the Brief Intro on Config System
            on <http://gitlab.bj.sensetime.com/xialiqiao/SenseStar>

        - cfg (:obj:`dict`): the config file, which is a multilevel dict, have
            large domain evaluate, common, model, train etc, and each large domain
            has it's smaller domain.

    Returns:
        - GradClip(:obj:`object`)the built gradient clipper
    """
    clip_type = cfg.train.grad_clip_type
    clip_value = cfg.train.grad_clip_value
    return GradClip(clip_value, clip_type)


class GradClip(object):
    r"""
    Overview:
        the gradient cliper

     Interface:
        __init__, apply
    """

    def __init__(self, clip_value, clip_type, norm_type=2):
        r"""
        Overview:
            initialization the GradClip.

            Note :
                reference <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

        Arguments:
            - clip_value (:obj:`float` or :obj:`int`): the defalut clip value
            - clip type (:obj:`str`): the clip type, now support max_norm and clip_value
        """
        assert (clip_type in ['max_norm', 'clip_value'])
        self.norm_type = norm_type
        self.clip_value = clip_value
        self.clip_type = clip_type

    def apply(self, parameters, value=None):
        r"""
        Overview:
            apply the GradClip to given parameters.

        Arguments:
            Note :
                reference <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

            - parameters (:obj:`Iterable[Tensor]` or :obj:`Tensor`): an iterable of Tensors or a single Tensor that will
                                                                     have gradients normalized
            - value (:obj:`float` or :obj:`int` or :obj:`None`): the clip value, if is None then use the defalut value

        """
        if value is None:
            v = self.clip_value
        else:
            v = value
        new_params = [t for t in parameters if t.requires_grad and t.grad is not None]
        if self.clip_type == 'max_norm':
            clip_grad_norm_(new_params, v, self.norm_type)
        elif self.clip_type == 'clip_value':
            clip_grad_value_(new_params, v)
