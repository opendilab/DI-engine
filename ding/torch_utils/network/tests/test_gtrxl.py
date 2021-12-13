import pytest
import torch

from ding.torch_utils import GTrXL


@pytest.mark.unittest
class TestGTrXL:

    def test(self):
        dim_size = 128
        seq_len = 64
        bs = 32
        action_dim = 4
        embedding_dim = 256
        layer_num = 5
        mem_len = 40
        # input shape: cur_seq x bs x input_dim
        memory = [None, torch.rand(layer_num, mem_len, bs, embedding_dim)]
        batch_first = [False, True]
        for i in range(2):
            m = memory[i]
            bf = batch_first[i]
            model = GTrXL(
                input_dim=dim_size,
                head_dim=2,
                embedding_dim=embedding_dim,
                memory_len=mem_len,
                head_num=2,
                mlp_num=2,
                layer_num=layer_num,
            )
            input = torch.rand(seq_len, bs, dim_size)
            if bf:
                input = torch.transpose(input, 1, 0)
            input.requires_grad_(True)
            if m is not None:
                model.reset(bs)
            output = model(input, batch_first=bf)
            loss = output['logit'].mean()
            loss.backward()
            assert isinstance(input.grad, torch.Tensor)
            if bf is False:
                assert output['logit'].shape == (seq_len, bs, embedding_dim)
            else:
                assert output['logit'].shape == (bs, seq_len, embedding_dim)
            assert output['memory'].shape == (layer_num+1, mem_len, bs, embedding_dim)
