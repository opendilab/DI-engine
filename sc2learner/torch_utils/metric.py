import torch
import random


def levenshtein_distance(pred, target):
    '''
    Overview: Levenshtein Distance(Edit Distance)
    Arguments:
        - (:obj:`torch.LongTensor`) pred, shape[N1]
        - (:obj:`torch.LongTensor`) target, shape[N2]
    Returns:
        - (:obj:`torch.LongTensor`) distance(scalar), shape[1]
    Note: N1 >= 0, N2 >= 0
    '''
    assert(isinstance(pred, torch.LongTensor) and isinstance(target, torch.LongTensor))
    assert(pred.device == target.device)
    N1, N2 = pred.shape[0], target.shape[0]
    assert(N1 >= 0 and N2 >= 0)
    if N1 == 0 or N2 == 0:
        distance = max(N1, N2)
    else:
        dp_array = torch.zeros(N1, N2).int()
        dp_array[0, :] = torch.arange(0, N2)
        dp_array[:, 0] = torch.arange(0, N1)
        for i in range(1, N1):
            for j in range(1, N2):
                if pred[i] == target[j]:
                    dp_array[i, j] = dp_array[i-1, j-1]
                else:
                    dp_array[i, j] = min(dp_array[i-1, j]+1, dp_array[i, j-1]+1, dp_array[i-1, j-1]+1)
        distance = dp_array[N1-1, N2-1]
    return torch.LongTensor([distance]).to(pred.device)


def hamming_distance(pred, target):
    '''
    Overview: Hamming Distance
    Arguments:
        - (:obj:`torch.LongTensor`) pred, shape[N]
        - (:obj:`torch.LongTensor`) target, shape[N]
    Returns:
        - (:obj:`torch.LongTensor`) distance(scalar), shape[1]
    Note: pred, target are also boolean vector(0 or 1)
    '''
    assert(isinstance(pred, torch.LongTensor) and isinstance(target, torch.LongTensor))
    assert(pred.device == target.device)
    assert(pred.shape == target.shape)
    return pred.ne(target).sum()


def test_levenshtein_distance():
    pred = torch.LongTensor([1, 4, 6, 4, 1])
    target1 = torch.LongTensor([1, 6, 4, 4, 1])
    distance = levenshtein_distance(pred, target1)
    assert(distance.item() == 2)

    target2 = torch.LongTensor([])
    distance = levenshtein_distance(pred, target2)
    assert(distance.item() == 5)

    target3 = torch.LongTensor([6, 4, 1])
    distance = levenshtein_distance(pred, target3)
    assert(distance.item() == 2)
    print('test_levenshtein_distance pass')


def test_hamming_distance():
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
        assert(distance.item() == diff)
    print('test_hamming_distance pass')


if __name__ == "__main__":
    test_levenshtein_distance()
    test_hamming_distance()
