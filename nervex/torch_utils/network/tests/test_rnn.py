import pytest
import torch
from nervex.torch_utils import get_lstm

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
        lstm_type = ['normal', 'pytorch']
        input = torch.rand(seq_len, batch_size, input_size).requires_grad_(True)
        for l in lstm_type:
            lstm = get_lstm(l, input_size, hidden_size, num_layers, norm_type, dropout)
            prev_state = None
            output, prev_state = lstm(input, prev_state, list_next_state=True)
            loss = output.mean()
            loss.backward()
            assert output.shape == (seq_len, batch_size, hidden_size)
            assert len(prev_state) == batch_size
            assert prev_state[0][0].shape == (num_layers, 1, hidden_size)
            assert isinstance(input.grad, torch.Tensor)

            prev_state = None
            for s in range(seq_len):
                input_step = input[s: s + 1]
                output, prev_state = lstm(input_step, prev_state, list_next_state=True)
            assert output.shape == (1, batch_size, hidden_size)
            assert len(prev_state) == batch_size
            assert prev_state[0][0].shape == (num_layers, 1, hidden_size)
            assert isinstance(input.grad, torch.Tensor)

            prev_state = None
            for s in range(seq_len):
                input_step = input[s: s + 1]
                output, prev_state = lstm(input_step, prev_state, list_next_state=False)
            assert output.shape == (1, batch_size, hidden_size)
            assert len(prev_state) == 2
            assert prev_state[0].shape == (num_layers, batch_size, hidden_size)
            assert isinstance(input.grad, torch.Tensor)