import random

import pytest
import torch

from ding.torch_utils.metric import levenshtein_distance, hamming_distance


@pytest.mark.unittest
class TestMetric():

    def test_levenshtein_distance(self):
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
        target3 = torch.LongTensor([6, 4, 1])
        distance = levenshtein_distance(pred, target3, pred, target3, extra_fn=lambda x, y: x + y)
        assert distance.item() == 13
        target4 = torch.LongTensor([1, 4, 1])
        distance = levenshtein_distance(pred, target4, pred, target4, extra_fn=lambda x, y: x + y)
        assert distance.item() == 14

    def test_hamming_distance(self):
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
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            distance = hamming_distance(pred, target)
            diff = len(set(pred_idx).union(set(target_idx)) - set(pred_idx).intersection(set(target_idx)))
            assert (distance.item() == diff)
