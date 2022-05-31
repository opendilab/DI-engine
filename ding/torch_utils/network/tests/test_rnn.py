import pytest
import torch
from ding.torch_utils import get_lstm, sequence_mask


@pytest.mark.unittest
class TestLstm:

    def test(self):
        seq_len = 2
        batch_size = 3
        input_size = 2
        hidden_size = 3
        num_layers = 2
        norm_type = 'LN'
        dropout = 0.1
        input = torch.rand(seq_len, batch_size, input_size).requires_grad_(True)
        # abnormal case
        lstm = get_lstm('normal', input_size, hidden_size, num_layers, norm_type, dropout)
        prev_state = torch.randn(4)
        with pytest.raises(TypeError):
            _, _ = lstm(input, prev_state, list_next_state=True)
        with pytest.raises(RuntimeError):
            _, _ = lstm(input, [[] for _ in range(batch_size + 1)], list_next_state=True)
        # normal case
        lstm_type = ['normal', 'pytorch']
        for l in lstm_type:
            lstm = get_lstm(l, input_size, hidden_size, num_layers, norm_type, dropout)
            prev_state = None
            output, prev_state = lstm(input, prev_state, list_next_state=True)
            loss = output.mean()
            loss.backward()
            assert output.shape == (seq_len, batch_size, hidden_size)
            assert len(prev_state) == batch_size
            assert prev_state[0]['h'].shape == (num_layers, 1, hidden_size)
            assert isinstance(input.grad, torch.Tensor)

            prev_state = None
            for s in range(seq_len):
                input_step = input[s:s + 1]
                output, prev_state = lstm(input_step, prev_state, list_next_state=True)
            assert output.shape == (1, batch_size, hidden_size)
            assert len(prev_state) == batch_size
            assert prev_state[0]['h'].shape == (num_layers, 1, hidden_size)
            assert isinstance(input.grad, torch.Tensor)

            prev_state = None
            for s in range(seq_len):
                input_step = input[s:s + 1]
                output, prev_state = lstm(input_step, prev_state, list_next_state=False)
            assert output.shape == (1, batch_size, hidden_size)
            assert len(prev_state) == 2
            assert prev_state['h'].shape == (num_layers, batch_size, hidden_size)
            assert isinstance(input.grad, torch.Tensor)

            randns = torch.randn(num_layers, 1, hidden_size)
            prev_state = [None for _ in range(batch_size)]
            prev_state[0] = {'h': randns, 'c': randns}
            output, prev_state = lstm(input, prev_state, list_next_state=True)


@pytest.mark.unittest
def test_sequence_mask():
    lengths = torch.LongTensor([0, 4, 3, 1, 2])
    masks = sequence_mask(lengths)
    assert masks.shape == (5, 4)
    assert masks.dtype == torch.bool
    masks = sequence_mask(lengths, max_len=3)
    assert masks.shape == (5, 3)
