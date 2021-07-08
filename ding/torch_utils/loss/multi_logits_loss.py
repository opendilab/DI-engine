import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.torch_utils.network import one_hot


def get_distance_matrix(lx, ly, mat, M: int) -> np.ndarray:
    nlx = np.broadcast_to(lx, [M, M]).T
    nly = np.broadcast_to(ly, [M, M])
    nret = nlx + nly - mat

    # ret = []
    # for i in range(M):
    #     ret.append(lx[i] + ly - mat[i])
    # ret = np.stack(ret)
    # assert ret.shape == (M, M)
    # assert np.all(nret == ret)
    return nret


class MultiLogitsLoss(nn.Module):
    '''
    Overview:
        Base class for supervised learning on linklink, including basic processes.
    Interface:
        forward
    '''

    def __init__(self, criterion: str = None, smooth_ratio: float = 0.1) -> None:
        '''
        Overview:
            initialization method, use cross_entropy as default criterion
        Arguments:
            - criterion (:obj:`str`): criterion type, supports ['cross_entropy', 'label_smooth_ce']
            - smooth_ratio (:obs:`float`): smooth_ratio for label smooth
        '''
        super(MultiLogitsLoss, self).__init__()
        if criterion is None:
            criterion = 'cross_entropy'
        assert (criterion in ['cross_entropy', 'label_smooth_ce'])
        self.criterion = criterion
        if self.criterion == 'label_smooth_ce':
            self.ratio = smooth_ratio

    def _label_process(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.LongTensor:
        N = logits.shape[1]
        if self.criterion == 'cross_entropy':
            return one_hot(labels, num=N)
        elif self.criterion == 'label_smooth_ce':
            val = float(self.ratio) / (N - 1)
            ret = torch.full_like(logits, val)
            ret.scatter_(1, labels.unsqueeze(1), 1 - val)
            return ret

    def _nll_loss(self, nlls: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        ret = (-nlls * (labels.detach()))
        return ret.sum(dim=1)

    def _get_metric_matrix(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        M, N = logits.shape
        labels = self._label_process(logits, labels)
        logits = F.log_softmax(logits, dim=1)
        metric = []
        for i in range(M):
            logit = logits[i]
            logit = logit.repeat(M).reshape(M, N)
            metric.append(self._nll_loss(logit, labels))
        return torch.stack(metric, dim=0)

    def _match(self, matrix: torch.Tensor):
        mat = matrix.clone().detach().to('cpu').numpy()
        mat = -mat  # maximize
        M = mat.shape[0]
        index = np.full(M, -1, dtype=np.int32)  # -1 note not find link
        lx = mat.max(axis=1)
        ly = np.zeros(M, dtype=np.float32)
        visx = np.zeros(M, dtype=np.bool)
        visy = np.zeros(M, dtype=np.bool)

        def has_augmented_path(t, binary_distance_matrix):
            # What is changed? visx, visy, distance_matrix, index
            # What is changed within this function? visx, visy, index
            visx[t] = True
            for i in range(M):
                if not visy[i] and binary_distance_matrix[t, i]:
                    visy[i] = True
                    if index[i] == -1 or has_augmented_path(index[i], binary_distance_matrix):
                        index[i] = t
                        return True
            return False

        for i in range(M):
            while True:
                visx.fill(False)
                visy.fill(False)
                distance_matrix = get_distance_matrix(lx, ly, mat, M)
                binary_distance_matrix = np.abs(distance_matrix) < 1e-4
                if has_augmented_path(i, binary_distance_matrix):
                    break
                masked_distance_matrix = distance_matrix[:, ~visy][visx]
                if 0 in masked_distance_matrix.shape:  # empty matrix
                    raise RuntimeError("match error, matrix: {}".format(matrix))
                else:
                    d = masked_distance_matrix.min()
                lx[visx] -= d
                ly[visy] += d
        return index

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        r"""
        Overview:
            Calculate multiple logits loss.
        Arguments:
            - logits (:obj:`torch.Tensor`): Predicted logits, whose shape must be 2-dim, like (B, N).
            - labels (:obj:`torch.LongTensor`): Ground truth.
        Returns:
            - loss (:obj:`torch.Tensor`): Calculated loss.
        """
        assert (len(logits.shape) == 2)
        metric_matrix = self._get_metric_matrix(logits, labels)
        index = self._match(metric_matrix)
        loss = []
        for i in range(metric_matrix.shape[0]):
            loss.append(metric_matrix[index[i], i])
        return sum(loss) / len(loss)
