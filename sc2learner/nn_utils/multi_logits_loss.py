import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sc2learner.nn_utils import one_hot


class MultiLogitsLoss(nn.Module):
    def __init__(self, criterion):
        super(MultiLogitsLoss, self).__init__()
        assert(criterion in ['cross_entropy'])
        self.criterion = criterion

    def _get_metric_matrix(self, logits, labels):
        M, N = logits.shape
        labels = one_hot(labels, num=N)
        logits = F.log_softmax(logits, dim=1)
        metric = []
        for i in range(M):
            logit = logits[i]
            logit = logit.repeat(M).reshape(M, N)
            metric.append(F.binary_cross_entropy_with_logits(logit, labels, reduction='none').sum(dim=1))
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
            visx[t] = True
            for i in range(M):
                if not visy[i] and lx[t] + ly[i] == mat[t, i]:
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
        assert(len(logits.shape) == 2)
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


if __name__ == "__main__":
    for _ in range(4):
        test_multi_logits_loss()
