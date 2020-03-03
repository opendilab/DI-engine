import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import LSTM


class CoreLstm(nn.Module):
    def __init__(self, cfg):
        super(CoreLstm, self).__init__()
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.lstm = LSTM(cfg.input_size, cfg.hidden_size, cfg.num_layers, norm_type='LN')

    def forward(self, embedded_entity, embedded_spatial, embedded_scalar, prev_state):
        '''
        Input:
            embedded_entity: [seq_len, batch_size, embed_dim_entity]
            embedded_spatial: [seq_len, batch_size, embed_dim_spatial]
            embedded_scalar: [seq_len, batch_size, embed_dim_scalar]
            prev_state: [num_layers, batch_size, hidden_size] * 2 or None
        '''
        embedded = torch.cat([embedded_entity, embedded_spatial, embedded_scalar], dim=2)
        output, next_state = self.lstm(embedded, prev_state, list_next_state=True)
        return output, next_state


def test_core_lstm():
    class CFG:
        def __init__(self):
            self.input_size = 256 + 256 + 640
            self.hidden_size = 384
            self.num_layers = 3

    model = CoreLstm(CFG()).cuda()
    B = 4
    S = 2
    embedded_entity = torch.randn(S, B, 256).cuda()
    embedded_spatial = torch.randn(S, B, 256).cuda()
    embedded_scalar = torch.randn(S, B, 640).cuda()
    prev_state = None
    output, next_state = model(embedded_entity, embedded_spatial, embedded_scalar, prev_state)
    print(output.shape)
    print(next_state[0].shape, next_state[1].shape)


if __name__ == "__main__":
    test_core_lstm()
