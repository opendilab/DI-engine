from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from ding.torch_utils.network.transformer import Attention
from ..common import ConvEncoder


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop_p=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_p)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, n_layer: int, n_attn: int, n_head: int, drop_p: float, max_T: int, n_ffn: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        assert n_attn % n_head == 0
        dim_head = n_attn // n_head
        for _ in range(n_layer):
            self.layers.append(nn.ModuleList([
                PreNorm(n_attn, Attention(n_attn, dim_head, n_attn, n_head, nn.Dropout(drop_p))),
                PreNorm(n_attn, FeedForward(n_attn, n_ffn, drop_p=drop_p))
            ]))
        self.mask = nn.Parameter(
            torch.tril(torch.ones((max_T, max_T), dtype=torch.bool)).view(1, 1, max_T, max_T), requires_grad=False
        )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, mask=self.mask) + x
            x = ff(x) + x
        return x


@MODEL_REGISTRY.register('pc_mcts')
class ProcedureCloningMCTS(nn.Module):

    def __init__(
        self,
        obs_shape: SequenceType,
        hidden_shape: SequenceType,
        action_dim: int,
        seq_len: int,
        cnn_hidden_list: SequenceType = [128, 256, 512],
        cnn_kernel_size: SequenceType = [8, 4, 3],
        cnn_stride: SequenceType = [4, 2, 1],
        cnn_padding: Optional[SequenceType] = [0, 0, 0],
        hidden_state_cnn_hidden_list: SequenceType = [128, 256, 512],
        hidden_state_cnn_kernel_size: SequenceType = [3, 3, 3],
        hidden_state_cnn_stride: SequenceType = [1, 1, 1],
        hidden_state_cnn_padding: Optional[SequenceType] = [1, 1, 1],
        cnn_activation: Optional[nn.Module] = nn.ReLU(),
        att_heads: int = 8,
        att_hidden: int = 512,
        n_att_layer: int = 4,
        ffn_hidden: int = 512,
        drop_p: float = 0.,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.hidden_shape = hidden_shape
        self.seq_len = seq_len
        max_T = seq_len + 1

        # Conv Encoder
        self.embed_state = ConvEncoder(
            obs_shape, cnn_hidden_list, cnn_activation, cnn_kernel_size, cnn_stride, cnn_padding
        )
        self.embed_hidden = ConvEncoder(
            hidden_shape, hidden_state_cnn_hidden_list, cnn_activation, hidden_state_cnn_kernel_size,
            hidden_state_cnn_stride, hidden_state_cnn_padding
        )

        self.cnn_hidden_list = cnn_hidden_list

        assert cnn_hidden_list[-1] == att_hidden
        self.transformer = Transformer(n_layer=n_att_layer, n_attn=att_hidden, n_head=att_heads,
                                       drop_p=drop_p, max_T=max_T, n_ffn=ffn_hidden)

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
        embedding_mask = torch.zeros(1, self.seq_len, 1).to(states.device)

        state_embeddings, hidden_state_embeddings = self._compute_embeddings(states, hidden_states)

        for i in range(self.seq_len):
            h = torch.cat((state_embeddings, hidden_state_embeddings * embedding_mask), dim=1)
            hidden_state_embeddings, action_pred = self._compute_transformer(h)
            embedding_mask[0, i, 0] = 1

        return action_pred
