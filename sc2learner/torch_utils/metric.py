"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Levenshtein_distance and hamming_distance: Calculate the levenshtein distance and the hamming distance 
        of the given inputs.
"""

import torch
import random


def levenshtein_distance(pred, target, pred_extra=None, target_extra=None, extra_fn=None):
    r"""
    Overview: 
        Levenshtein Distance(Edit Distance)
    
    Arguments:
        Note: 
            N1 >= 0, N2 >= 0

        - pred (:obj:`torch.LongTensor`): shape[N1]
        - target (:obj:`torch.LongTensor`): shape[N2]
        - pred_extra (:obj:`torch.Tensor or None`)
        - target_extra (:obj:`torch.Tensor or None`)
        - extra_fn (:obj:`function or None`): if specified, the distance metric of the extra input data
    
    Returns:
        - (:obj:`torch.FloatTensor`) distance(scalar), shape[1]
    """
    assert (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor))
    assert (pred.dtype == torch.long and target.dtype == torch.long), '{}\t{}'.format(pred.dtype, target.dtype)
    assert (pred.device == target.device)
    assert (type(pred_extra) == type(target_extra))
    if not extra_fn:
        assert (not pred_extra)
    N1, N2 = pred.shape[0], target.shape[0]
    assert (N1 >= 0 and N2 >= 0)
    if N1 == 0 or N2 == 0:
        distance = max(N1, N2)
    else:
        dp_array = torch.zeros(N1, N2).float()
        if extra_fn:
            if pred[0] == target[0]:
                extra = extra_fn(pred_extra[0], target_extra[0])
            else:
                extra = 1.
            dp_array[0, :] = torch.arange(0, N2) + extra
            dp_array[:, 0] = torch.arange(0, N1) + extra
        else:
            dp_array[0, :] = torch.arange(0, N2)
            dp_array[:, 0] = torch.arange(0, N1)
        for i in range(1, N1):
            for j in range(1, N2):
                if pred[i] == target[j]:
                    if extra_fn:
                        dp_array[i, j] = dp_array[i - 1, j - 1] + extra_fn(pred_extra[i], target_extra[j])
                    else:
                        dp_array[i, j] = dp_array[i - 1, j - 1]
                else:
                    dp_array[i, j] = min(dp_array[i - 1, j] + 1, dp_array[i, j - 1] + 1, dp_array[i - 1, j - 1] + 1)
        distance = dp_array[N1 - 1, N2 - 1]
    return torch.FloatTensor([distance]).to(pred.device)


def hamming_distance(pred, target, weight=1.):
    r'''
    Overview:
        Hamming Distance

    Arguments:
        Note:
            pred, target are also boolean vector(0 or 1)

        - pred (:obj:`torch.LongTensor`): pred input, shape[B, N], while B is the batch size
        - target (:obj:`torch.LongTensor`): target input, shape[B, N], while B is the batch size
    
    Returns:
        - distance(:obj:`torch.LongTensor`): distance(scalar), the shape[1]
    
    Shapes:
        - pred & targe (:obj:`torch.LongTensor`): shape :math:`(B, N)`, while B is the batch size and N is the dimension
    '''
    assert (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor))
    assert (pred.dtype == torch.long and target.dtype == torch.long)
    assert (pred.device == target.device)
    assert (pred.shape == target.shape)
    return pred.ne(target).sum(dim=1).float().mul_(weight)

#TODO
#集成到pytest
def test_levenshtein_distance():
    r'''
    Overview: 
        Test the Levenshtein Distance
    '''
    pred = torch.LongTensor([1, 4, 6, 4, 1])
    target1 = torch.LongTensor([1, 6, 4, 4, 1])
    distance = levenshtein_distance(pred, target1)
    assert (distance.item() == 2)

    target2 = torch.LongTensor([])
    distance = levenshtein_distance(pred, target2)
    assert (distance.item() == 5)

    target3 = torch.LongTensor([6, 4, 1])
    distance = levenshtein_distance(pred, target3)
    assert (distance.item() == 2)
    print('test_levenshtein_distance pass')

#TODO
#集成到pytest
def test_hamming_distance():
    r'''
    Overview: 
        Test the Hamming Distance
    '''
    base = torch.zeros(8).long()
    index = [i for i in range(8)]
    for i in range(2):
        pred_idx = random.sample(index, 4)
        target_idx = random.sample(index, 4)
        pred = base.clone()
        pred[pred_idx] = 1
        target = base.clone()
        target[target_idx] = 1
        distance = hamming_distance(pred, target)
        diff = len(set(pred_idx).union(set(target_idx)) - set(pred_idx).intersection(set(target_idx)))
        assert (distance.item() == diff)
    print('test_hamming_distance pass')


if __name__ == "__main__":
    test_levenshtein_distance()
    test_hamming_distance()
