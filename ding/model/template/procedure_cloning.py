from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from ding.torch_utils.network.transformer import Attention
from ding.torch_utils.network.nn_module import fc_block, build_normalization
from ..common import FCEncoder, ConvEncoder


class Block(nn.Module):

    def __init__(
            self, cnn_hidden: int, att_hidden: int, att_heads: int, drop_p: float, max_T: int, n_att: int,
            feedforward_hidden: int, n_feedforward: int
    ) -> None:
        super().__init__()
        self.n_att = n_att
        self.n_feedforward = n_feedforward
        self.attention_layer = []

        self.norm_layer = [nn.LayerNorm(att_hidden)] * n_att
        self.attention_layer.append(Attention(cnn_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
        for i in range(n_att - 1):
            self.attention_layer.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))

        self.att_drop = nn.Dropout(drop_p)

        self.fc_blocks = []
        self.fc_blocks.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
        for i in range(n_feedforward - 1):
            self.fc_blocks.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
        self.norm_layer.extend([nn.LayerNorm(feedforward_hidden)] * n_feedforward)
        self.mask = torch.tril(torch.ones((max_T, max_T), dtype=torch.bool)).view(1, 1, max_T, max_T)

    def forward(self, x: torch.Tensor):
        for i in range(self.n_att):
            x = self.att_drop(self.attention_layer[i](x, self.mask))
            x = self.norm_layer[i](x)
        for i in range(self.n_feedforward):
            x = self.fc_blocks[i](x)
            x = self.norm_layer[i + self.n_att](x)
        return x


@MODEL_REGISTRY.register('pc')
class ProcedureCloning(nn.Module):

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_shape: SequenceType,
            action_dim: int,
            seq_len: int,
            cnn_hidden_list: SequenceType = [128, 128, 256, 256, 256],
            cnn_activation: Optional[nn.Module] = nn.ReLU(),
            cnn_kernel_size: SequenceType = [3, 3, 3, 3, 3],
            cnn_stride: SequenceType = [1, 1, 1, 1, 1],
            cnn_padding: Optional[SequenceType] = [1, 1, 1, 1, 1],
            mlp_hidden_list: SequenceType = [256, 256],
            mlp_activation: Optional[nn.Module] = nn.ReLU(),
            att_heads: int = 8,
            att_hidden: int = 128,
            n_att: int = 4,
            n_feedforward: int = 2,
            feedforward_hidden: int = 256,
            drop_p: float = 0.5,
            augment: bool = True,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.hidden_shape = hidden_shape
        self.seq_len = seq_len
        max_T = seq_len + 1

        #Conv Encoder
        print(cnn_padding)
        self.embed_state = ConvEncoder(
            obs_shape, cnn_hidden_list, cnn_activation, cnn_kernel_size, cnn_stride, cnn_padding
        )
        self.embed_hidden = ConvEncoder(
            hidden_shape, cnn_hidden_list, cnn_activation, cnn_kernel_size, cnn_stride, cnn_padding
        )

        self.cnn_hidden_list = cnn_hidden_list
        self.augment = augment

        assert cnn_hidden_list[-1] == mlp_hidden_list[-1]
        layers = []
        for i in range(n_att):
            if i == 0:
                layers.append(Attention(cnn_hidden_list[-1], att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            else:
                layers.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            layers.append(build_normalization('LN')(att_hidden))
        for i in range(n_feedforward):
            if i == 0:
                layers.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
            else:
                layers.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
                self.layernorm2 = build_normalization('LN')(feedforward_hidden)

        self.transformer = Block(
            cnn_hidden_list[-1], att_hidden, att_heads, drop_p, max_T, n_att, feedforward_hidden, n_feedforward
        )

        self.predict_hidden_state = torch.nn.Linear(cnn_hidden_list[-1], cnn_hidden_list[-1])
        self.predict_action = torch.nn.Linear(cnn_hidden_list[-1], action_dim)

    def _compute_embeddings(self, states: torch.Tensor, hidden_states: torch.Tensor):
        B, T, *_ = hidden_states.shape

        # shape: (B, 1, h_dim)
        state_embeddings = self.embed_state(states).reshape(B, 1, self.cnn_hidden_list[-1])
        # shape: (B, T, h_dim)
        hidden_state_embeddings = self.embed_hidden(hidden_states.reshape(B * T, *hidden_states.shape[2:])) \
            .reshape(B, T, self.cnn_hidden_list[-1])
        return state_embeddings, hidden_state_embeddings

    def _compute_transformer(self, h):
        B, T, *_ = h.shape
        h = self.transformer(h)
        h = h.reshape(B, T, self.cnn_hidden_list[-1])

        hidden_state_preds = self.predict_hidden_state(h[:, 0:-1, ...])
        action_preds = self.predict_action(h[:, -1, :])
        return hidden_state_preds, action_preds

    def forward(self, states: torch.Tensor, hidden_states: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # State is current observation.
        # Hidden states is a sequence including [L, R, ...].
        # The shape of state and hidden state may be different.
        B, T, *_ = hidden_states.shape
        assert T == self.seq_len
        state_embeddings, hidden_state_embeddings = self._compute_embeddings(states, hidden_states)

        h = torch.cat((state_embeddings, hidden_state_embeddings), dim=1)
        hidden_state_preds, action_preds = self._compute_transformer(h)

        return hidden_state_preds, action_preds, hidden_state_embeddings.detach()

    def forward_eval(self, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        hidden_states = torch.zeros(batch_size, self.seq_len, *self.hidden_shape, dtype=states.dtype).to(states.device)
        embedding_mask = torch.zeros(1, self.seq_len, 1)

        state_embeddings, hidden_state_embeddings = self._compute_embeddings(states, hidden_states)

        for i in range(self.seq_len):
            h = torch.cat((state_embeddings, hidden_state_embeddings * embedding_mask), dim=1)
            hidden_state_embeddings, action_pred = self._compute_transformer(h)
            embedding_mask[0, i, 0] = 1

        return action_pred
