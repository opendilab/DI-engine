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


def build_criterion(cfg):
    if cfg.type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif cfg.type == 'label_smooth_ce':
        return LabelSmoothCELoss(cfg.kwargs.smooth_ratio)
    else:
        raise ValueError("invalid criterion type:{}".format(cfg.type))


def test_label_smooth_ce_loss():
    logits = torch.randn(4, 6)
    labels = torch.LongTensor([i for i in range(4)])
    criterion1 = LabelSmoothCELoss(0)
    criterion2 = nn.CrossEntropyLoss()
    print(criterion1(logits, labels))
    print(criterion2(logits, labels))
    assert (torch.abs(criterion1(logits, labels) - criterion2(logits, labels)) < 1e-6)
    print("test end")


if __name__ == "__main__":
    test_label_smooth_ce_loss()
