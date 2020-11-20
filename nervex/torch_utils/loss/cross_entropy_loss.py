import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCELoss(nn.Module):

    def __init__(self, ratio):
        super(LabelSmoothCELoss, self).__init__()
        self.ratio = ratio

    def forward(self, logits, labels):
        B, N = logits.shape
        val = float(self.ratio) / (N - 1)
        one_hot = torch.full_like(logits, val)
        one_hot.scatter_(1, labels.unsqueeze(1), 1 - val)
        logits = F.log_softmax(logits, dim=1)
        return -torch.sum(logits * (one_hot.detach())) / B


class SoftFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True, reduce=None):
        super(SoftFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average, reduce=reduce)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs, 1)) ** self.gamma * F.log_softmax(inputs, 1), targets)


def build_ce_criterion(cfg):
    if cfg.type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif cfg.type == 'label_smooth_ce':
        return LabelSmoothCELoss(cfg.kwargs.smooth_ratio)
    elif cfg.type == 'soft_focal_loss':
        return SoftFocalLoss()
    else:
        raise ValueError("invalid criterion type:{}".format(cfg.type))
