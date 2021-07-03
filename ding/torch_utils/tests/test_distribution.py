import pytest
import torch

from ding.torch_utils.distribution import Pd, CategoricalPd, CategoricalPdPytorch


@pytest.mark.unittest
class TestProbDistribution:

    def test_Pd(self):
        pd = Pd()
        with pytest.raises(NotImplementedError):
            pd.neglogp(torch.randn(5, ))
        with pytest.raises(NotImplementedError):
            pd.noise_mode()
        with pytest.raises(NotImplementedError):
            pd.mode()
        with pytest.raises(NotImplementedError):
            pd.sample()

    def test_CatePD(self):
        pd = CategoricalPd()
        logit1 = torch.randn(3, 5, requires_grad=True)
        logit2 = torch.randint(5, (3, ), dtype=torch.int64)

        pd.update_logits(logit1)
        entropy = pd.neglogp(logit2)
        assert entropy.requires_grad
        assert entropy.shape == torch.Size([])

        entropy = pd.entropy()
        assert entropy.requires_grad
        assert entropy.shape == torch.Size([])
        entropy = pd.entropy(reduction=None)
        assert entropy.requires_grad
        assert entropy.shape == torch.Size([3])

        ret = pd.sample()
        assert ret.shape == torch.Size([3])
        ret = pd.sample(viz=True)
        assert ret[0].shape == torch.Size([3])

        ret = pd.mode()
        assert ret.shape == torch.Size([3])
        ret = pd.mode(viz=True)
        assert ret[0].shape == torch.Size([3])

        ret = pd.noise_mode()
        assert ret.shape == torch.Size([3])
        ret = pd.noise_mode(viz=True)
        assert ret[0].shape == torch.Size([3])

        pd = CategoricalPdPytorch()
        pd.update_logits(logit1)

        ret = pd.sample()
        assert ret.shape == torch.Size([3])
        ret = pd.mode()
        assert ret.shape == torch.Size([3])

        entropy = pd.entropy(reduction='mean')
        assert entropy.requires_grad
        assert entropy.shape == torch.Size([])
        entropy = pd.entropy(reduction=None)
        assert entropy.requires_grad
        assert entropy.shape == torch.Size([3])
