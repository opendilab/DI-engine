'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. implementation of MultiLogitsLoss and its test
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sc2learner.torch_utils.network import one_hot


class MultiLogitsLoss(nn.Module):
    '''
        Overview: base class for supervised learning on linklink, including basic processes.
        Interface: __init__, forward
    '''

    def __init__(self, cfg=None, criterion=None, smooth_ratio=0.1):
        '''
            Overview: initialization method, use cross_entropy as default criterion
            Arguments:
                - cfg (:obj:`dict`): config(type and kwargs), if cfg is not None, first use cfg
                - criterion (:obj:`str`): criterion type
                - smooth_ratio (:obs:`float`): smooth_ratio for label smooth
        '''
        super(MultiLogitsLoss, self).__init__()
        if cfg is not None:
            criterion = cfg.type
            assert (criterion in ['cross_entropy', 'label_smooth_ce'])
            if criterion == 'label_smooth_ce':
                smooth_ratio = cfg.kwargs['smooth_ratio']
        self.criterion = criterion
        if self.criterion == 'label_smooth_ce':
            self.ratio = smooth_ratio

    def _label_process(self, labels, logits):
        N = logits.shape[1]
        if self.criterion == 'cross_entropy':
            return one_hot(labels, num=N)
        elif self.criterion == 'label_smooth_ce':
            val = float(self.ratio) / (N - 1)
            ret = torch.full_like(logits, val)
            ret.scatter_(1, labels.unsqueeze(1), 1 - val)
            return ret

    def _nll_loss(self, nlls, labels):
        ret = (-nlls * (labels.detach()))
        return ret.sum(dim=1)

    def _get_metric_matrix(self, logits, labels):
        M, N = logits.shape
        labels = self._label_process(labels, logits)
        logits = F.log_softmax(logits, dim=1)
        metric = []
        for i in range(M):
            logit = logits[i]
            logit = logit.repeat(M).reshape(M, N)
            metric.append(self._nll_loss(logit, labels))
        return torch.stack(metric, dim=0)

    def _match(self, matrix):
        mat = matrix.clone().detach().to('cpu').numpy()
        mat = -mat  # maximize
        M, _ = mat.shape
        index = np.full(M, -1, dtype=np.int32)  # -1 note not find link
        lx = mat.max(axis=1)
        ly = np.zeros(M, dtype=np.float32)
        visx = np.zeros(M, dtype=np.bool)
        visy = np.zeros(M, dtype=np.bool)

        def has_augmented_path(t):
            # FIXME this function take extremely long time (7% of the total running time)
            visx[t] = True
            for i in range(M):
                if not visy[i] and math.fabs(lx[t] + ly[i] - mat[t, i]) < 1e-4:
                    visy[i] = True
                    if index[i] == -1 or has_augmented_path(index[i]):
                        index[i] = t
                        return True
            return False

        for i in range(M):
            while True:
                visx = np.zeros(M, dtype=np.bool)
                visy = np.zeros(M, dtype=np.bool)
                if has_augmented_path(i):
                    break
                d = np.inf
                for j in range(M):
                    if visx[j]:
                        for k in range(M):
                            if not visy[k]:
                                d = min(d, lx[j] + ly[k] - mat[j, k])
                if d == np.inf:
                    raise Exception("match error, matrix: {}".format(matrix))
                for j in range(M):
                    if visx[j]:
                        lx[j] -= d
                    if visy[j]:
                        ly[j] += d
        return index

    def forward(self, logits, labels):
        assert (len(logits.shape) == 2)
        metric_matrix = self._get_metric_matrix(logits, labels)
        index = self._match(metric_matrix)
        loss = []
        for i in range(metric_matrix.shape[0]):
            loss.append(metric_matrix[index[i], i])
        return sum(loss) / len(loss)


def test_multi_logits_loss():
    logits = torch.randn(4, 8).cuda()
    label = torch.LongTensor([0, 1, 3, 2]).cuda()
    loss = MultiLogitsLoss(criterion='cross_entropy').cuda()
    print(loss(logits, label))


def _selected_units_loss():
    def smooth_label(label, num, eps=0.1):
        val = eps / (num - 1)
        ret = torch.full((1, num), val)
        ret[0, label] = 1 - eps
        return ret

    logits = torch.load('logits.pt')
    label = torch.load('labels.pt')
    criterion = MultiLogitsLoss(criterion='cross_entropy')
    self_criterion = nn.CrossEntropyLoss()
    label = [x for x in label if isinstance(x, torch.Tensor)]
    print(len(label))
    print(label)
    if len(label) == 0:
        return 0
    loss = []
    for b in range(len(label)):
        lo, la = logits[b], label[b]
        lo = torch.cat(lo, dim=0)
        print(b, lo.shape, la.shape)
        if lo.shape[0] != la.shape[0]:
            assert (lo.shape[0] == 1 + la.shape[0])
            end_flag_label = torch.LongTensor([lo.shape[1] - 1]).to(la.device)
            # end_flag_label = smooth_label(lo.shape[1]-1, lo.shape[1]).to(la.device)
            # end_flag_loss = F.binary_cross_entropy_with_logits(lo[-1:], end_flag_label)
            end_flag_loss = self_criterion(lo[-1:], end_flag_label)
            logits_loss = criterion(lo[:-1], la)
            print(end_flag_loss, logits_loss)
            loss.append((end_flag_loss + logits_loss) / 2)
        else:
            loss.append(criterion(lo, la))
    print(sum(loss) / len(loss), len(label))


if __name__ == "__main__":
    for _ in range(4):
        test_multi_logits_loss()
    # _selected_units_loss()
