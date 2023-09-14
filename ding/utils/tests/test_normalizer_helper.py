import easydict
import numpy
import pytest

from ding.utils.normalizer_helper import DatasetNormalizer

@pytest.mark.unittest
class TestNormalizerHelper:
    def test_normalizer(self):
        x = numpy.random.randn(10)
        mean = x.mean()
        std = x.std()
        normalizer = DatasetNormalizer({'test': x}, 'GaussianNormalizer', 10)
        test = numpy.random.randn(1)
        normal_test = normalizer.normalize(test, 'test')
        unnormal_test =normalizer.unnormalize(test, 'test')
        assert unnormal_test == test
        assert normal_test == (test - mean) / std
        